#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan, Imu, Image
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import TwistStamped, PoseStamped
from std_msgs.msg import Float64MultiArray


from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy import ndimage
import math
import heapq
from enum import Enum


class MissionFSM(Node):

    def _publish_cmd(self, lx, az):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'   # convention; controller doesn't enforce
        msg.twist.linear.x = lx
        msg.twist.angular.z = az
        self.cmd_vel_pub.publish(msg)

    def _publish_gripper(self, data):
        # Garra: [elevação, braço dir, braço esq] em metros. O controlador SEGURA a última
        # pose, então republicar a cada tick é idempotente e robusto a perdas de msg.
        msg = Float64MultiArray()
        msg.data = list(data)
        self.gripper_pub.publish(msg)

    class States(Enum):
        IDLE = 0
        EXPLORING = 1
        OBSTACLE_AVOIDANCE = 2
        FLAG_FOUND_CONFIRMATION = 3
        GOING_TO_FLAG = 4
        REFINDING_FLAG = 5
        ADJUSTING_POSITION_TO_COLLECT_FLAG = 6
        COLLECTING_FLAG = 7
        RETURNING_HOME = 8

    def behavior_idle(self):
        self._publish_cmd(0.0, 0.0)


    def behavior_exploring(self):
        # Busca DIRIGIDA: navega para o waypoint na zona azul (onde a bandeira fixa
        # está), reusando o controlador go-to-point. Enquanto a 1ª msg de odom não
        # chega (sem pose/waypoint), cai para o avanço reto antigo — sem regressão.
        if self.pose is None or self.search_goal is None:
            self._publish_cmd(0.9, 0.0)
            return
        self._follow_path(*self.search_goal)


    def behavior_obstacle_avoidance(self):
        # Desvio em TRÊS fases para escapar de mínimo local (obstáculo entre robô e
        # meta): (1) BACKUP dá ré se a frente está perto demais para girar com o braço
        # sem raspar; (2) TURN gira no lugar para o lado escolhido; (3) ESCAPE anda PARA
        # FRENTE nessa direção, deslocando-se LATERALMENTE ao redor do obstáculo.
        if self.avoid_phase == 'BACKUP':   # dá ré p/ afastar o braço antes de girar
            self._publish_cmd(-self.AVOID_BACK, 0.0)
        elif self.avoid_phase == 'TURN':
            self._publish_cmd(0.0, self.AVOID_ANGULAR * self.avoid_turn_dir)
        else:  # ESCAPE: anda em ARCO, curvando de volta ao lado do obstáculo (contorna).
            self._publish_cmd(self.ESCAPE_FWD, -self.avoid_turn_dir * self.ESCAPE_CURVE)

        # Quando estiver carregando a bandeira, continua republicando a pose levantada
        # para manter o gripper no lugar enquanto desvia.
        if self._is_carrying():
            self._publish_gripper(self.GRIPPER_LIFTED)


    def behavior_flag_found_confirmation(self):
        self._publish_cmd(0.0, 0.0)


    def behavior_going_to_flag(self):
        # Navega para a META LEMBRADA da bandeira (frame odom), NÃO para o pixel atual.
        # Assim, perder a bandeira de vista (ex.: durante um desvio) não nos faz parar:
        # seguimos para o ponto memorizado; a câmera só refina a meta quando a vê.
        if self.flag_goal is None:
            self._publish_cmd(0.0, 0.0)   # sem meta ainda; a transição trata (REFINDING)
            return
        dist = self._follow_path(*self.flag_goal)
        self.get_logger().info(
            f"GOING_TO_FLAG dist={dist:.2f} area={self.latest_flag_area} "
            f"vis={self.latest_flag_centroid is not None}",
            throttle_duration_sec=0.5)


    def behavior_refinding_flag(self):
        # Gira no lugar para o lado onde a bandeira deve estar. Melhor estimativa de
        # direção = a META LEMBRADA no frame odom (_flag_side: +1 esq / -1 dir), que
        # vale qualquer que tenha sido o caminho de entrada no REFINDING. Sem meta/pose
        # (ou bandeira ~à frente), cai para o último erro de pixel visto (last_flag_err):
        # err<=0 (esquerda) → CCW(+); err>0 (direita) → CW(-).
        side = self._flag_side()
        if side is not None:
            sign = side
        else:
            sign = 1.0 if self.last_flag_err <= 0 else -1.0
        self._publish_cmd(0.0, self.REFIND_ANGULAR * sign)


    def behavior_adjusting_position(self):
        # Já chegou perto (AREA_NEAR). Agora só GIRA NO LUGAR (linear=0) para deixar
        # a bandeira centralizada no quadro antes de "coletar". Mesmo erro de bearing
        # do GOING_TO_FLAG, mas sem avançar — ajuste fino de orientação.
        centroid = self.latest_flag_centroid
        if centroid is None or self.image_width is None:
            self._publish_cmd(0.0, 0.0)  # sem alvo; a transição manda p/ REFINDING
            return
        cx, _ = centroid
        err = self._flag_bearing_error(cx)     # ∈ [-1, +1]; + = bandeira à direita
        self.last_flag_err = err               # mantém o lado p/ REFINDING girar certo
        angular = -self.KP_BEARING * err       # gira para zerar o erro de bearing
        self._publish_cmd(0.0, angular)
        self.get_logger().info(
            f"ADJUSTING err={err:+.2f} area={self.latest_flag_area:.0f}",
            throttle_duration_sec=0.5)


    def behavior_collecting_flag(self):
        # Captura em sub-fases. O comportamento ATUA (cmd_vel + garra) conforme a fase;
        # o avanço/fecho de fase e a saída vivem na transição (divisão comportamento×
        # transição do FSM). A garra segura a pose → republicar a cada tick é seguro.
        if self.collect_phase == 'OPEN':
            self._publish_cmd(0.0, 0.0)            # parado enquanto a garra abre
            self._publish_gripper(self.GRIPPER_OPEN)
        elif self.collect_phase == 'CREEP':
            self._publish_gripper(self.GRIPPER_OPEN)   # mantém aberta enquanto avança
            # Centra no MASTRO via LIDAR (não no centroide da câmera): o centroide do
            # blob inclui o PAINEL deslocado ~0.16 m do mastro → centrar nele encosta a
            # garra ao LADO. O feixe LIDAR mais próximo à frente aponta para o mastro.
            bearing = self._front_min_bearing()
            ang = 0.0 if bearing is None else self.CREEP_KP_ANG * bearing  # +bearing=esq→CCW(+)
            self._publish_cmd(self.CREEP_FWD, ang)
        else:  # CLOSE
            self._publish_cmd(0.0, 0.0)            # parado: fecha e deixa o aperto assentar
            self._publish_gripper(self.GRIPPER_CLOSED)
        _b = self._front_min_bearing()
        self.get_logger().info(
            f"COLLECTING phase={self.collect_phase} front={self._front_min_dist():.2f} "
            f"bearing={'na' if _b is None else f'{math.degrees(_b):+.0f}'} "
            f"area={self.latest_flag_area}", throttle_duration_sec=0.5)


    def behavior_returning_home(self):
        # Retorno à base em sub-fases: LIFT (levanta braço) → NAVIGATE (vai p/ home) →
        # DEPOSIT (abaixa + abre a garra; espera 2 s parado p/ a bandeira assentar) →
        # REVERSING (ré lenta de 0.5 m p/ afastar a garra da bandeira) →
        # CLOSING (fecha a garra) → DONE (parado, missão encerrada).
        if self.return_phase == 'DEPOSIT':
            # Comando à garra com elevação 0 + pinças abertas: ABAIXA o braço (estava
            # levantado) e ABRE → deposita a bandeira no chão à frente do robô. Parado.
            self._publish_cmd(0.0, 0.0)
            self._publish_gripper(self.GRIPPER_OPEN)
            return
        if self.return_phase == 'REVERSING':
            # Ré lenta p/ recuar 0.5 m, garra ainda ABERTA (não arrasta a bandeira de volta).
            self._publish_cmd(-self.REVERSE_SPEED, 0.0)
            self._publish_gripper(self.GRIPPER_OPEN)
            return
        if self.return_phase == 'CLOSING':
            # Fecha a garra, parado — gesto final antes de encerrar a missão.
            self._publish_cmd(0.0, 0.0)
            self._publish_gripper(self.GRIPPER_CLOSED)
            return
        if self.return_phase == 'DONE':
            # Missão concluída: garra fechada, robô PARADO. Estado terminal (sem transição).
            self._publish_cmd(0.0, 0.0)
            self._publish_gripper(self.GRIPPER_CLOSED)
            return

        # LIFT/NAVIGATE: a garra SEGURA a pose levantada (idempotente, padrão de COLLECTING).
        self._publish_gripper(self.GRIPPER_LIFTED)
        if self.return_phase == 'LIFT':
            self._publish_cmd(0.0, 0.0)       # parado enquanto o braço sobe
            return

        # Fase NAVIGATE: dirige até o ponto de spawn (home).
        if self.pose is None or self.start_pose is None:
            self._publish_cmd(0.0, 0.0)
            return
        home = (self.start_pose[0], self.start_pose[1])
        dist = self._follow_path(*home)
        self.get_logger().info(
            f"RETURNING_HOME dist={dist:.2f}", throttle_duration_sec=0.5)


    # --- Percepção: helpers de LIDAR ---
    # LIDAR (via `ros2 topic echo /scan`): angle_min=0, ~1°/raio, range=[0.12, 3.5],
    # índice 0 = FRENTE. Ranges medidos da origem do sensor (x=0), não da ponta do braço.
    FRONT_ARC_HALF = math.radians(25.0)   # meia-largura do arco frontal (rad)
    # Braço/garra vai até x≈0.40 → folga real na ponta = r − 0.40. Bloqueio com ~0.15 m
    # de margem na ponta + reação ⇒ r ≤ ~0.65 m (LIDAR).
    FRONT_BLOCK_DIST = 0.7                # m da origem do LIDAR
    # PERIGO: mais apertado que "bloqueado". Na fase ESCAPE, se algo fica perigosamente
    # perto à frente, volta a dar RÉ em vez de girar no lugar (o braço raspa).
    DANGER_DIST = 0.5                     # m (~0.1 m da ponta do braço)

    # OBSTACLE_AVOIDANCE — desvio COMPROMETIDO (anti-chatter), camada subsumption.
    AVOID_ANGULAR = 0.5                   # rad/s do giro (sentido travado na entrada)
    # Histerese (Schmitt): entra em FRONT_BLOCK_DIST, mas só sai com folga MAIOR
    # (CLEAR_DIST). Limiares iguais = vibração na fronteira.
    CLEAR_DIST = 1.0                      # m: frente livre até aqui p/ sair do desvio
    AVOID_MIN_TICKS = 10                  # gira ≥1 s antes de cogitar sair (mata chatter)
    # Anti-livelock: se girou ~6 s sem achar folga LARGA, RELAXA p/ FRONT_BLOCK_DIST —
    # aceita qualquer saída não-perigosa em vez de girar p/ sempre.
    AVOID_MAX_TICKS = 60
    # ESCAPE (anti-mínimo-local): após liberar a frente, anda PARA FRENTE arqueando de
    # volta ao lado do obstáculo (-avoid_turn_dir) → CONTORNA em arco em vez de orbitar.
    ESCAPE_FWD = 0.35           # m/s à frente no escape
    ESCAPE_TICKS = 20           # ~1.5 s ≈ 0.37 m de deslocamento lateral
    ESCAPE_CURVE = 0.2          # rad/s de curva (menor → arco mais aberto; raio ~0.6 m)
    # BACKUP: girar no lugar perto de um obstáculo faz o braço (0.4 m) raspar; se a
    # frente está perto demais p/ girar, dá RÉ primeiro p/ abrir espaço.
    ROTATE_SAFE_DIST = 0.6      # m: frente livre até aqui = seguro girar
    AVOID_BACK = 0.15           # m/s de ré no BACKUP
    BACKUP_MAX_TICKS = 15       # teto de ré (~0.22 m)

    # Confirmação da bandeira: ticks consecutivos visível antes de aceitar (~2 s a
    # 10 Hz). Conta TICKS (taxa fixa) = tempo real, não frames da câmera.
    CONFIRM_TICKS = 20

    # COLLECTING_FLAG — captura REAL em SUB-FASES (espelha o OBSTACLE_AVOIDANCE):
    # OPEN (abre parado) → CREEP (avança centralizando no mastro) → CLOSE (fecha/aperta).
    GRIPPER_OPEN = [0.0, -0.06, 0.06]   # [elevação, dir, esq]: pinças abertas ±6 cm
    GRIPPER_CLOSED = [0.0, 0.0, 0.0]    # fechadas (fenda ~2 cm) → aperta o mastro de 6 cm
    GRIPPER_LIFTED = [-0.5, 0.0, 0.0]   # braço UP ~28°, fechado, haste acima do LIDAR
    OPEN_TICKS = 10        # ~1 s p/ as pinças abrirem antes de avançar
    CREEP_FWD = 0.08       # m/s: avanço lento na aproximação final
    # PREENSÃO via LIDAR: o mastro cruza o plano do LIDAR (z=0.12); a garra aberta em
    # ext=0 fica abaixo dele → _front_min_dist lê o mastro limpo. Superfície a ~0.40 m
    # quando entre as pinças (pontas x≈0.43). TUNAR vendo front_min no log.
    GRASP_DIST = 0.40      # m: para de avançar e FECHA quando o mastro chega aqui
    CREEP_MAX_TICKS = 120  # teto de segurança (~0.95 m); o LIDAR encerra o creep antes
    CLOSE_TICKS = 15       # ~1.5 s p/ o aperto assentar antes de declarar captura
    CREEP_KP_ANG = 1.5     # rad/s por rad de bearing do mastro (LIDAR) → centra no mastro

    # RETURNING_HOME — retorno à base carregando a bandeira (corpo ampliado).
    CARRY_BODY_EXTRA = 0.2     # m: o mastro estende a frente; soma-se aos limiares de desvio
    CARRY_ARM_HALF_DEG = 15.0  # °: zona-morta no centro do LIDAR p/ ignorar a garra carregada
    CARRY_W_MAX = 0.3          # rad/s: giro máx ao carregar (mais lento)
    CARRY_V_MAX = 0.20         # m/s: velocidade máx ao carregar (mais cuidado)
    HOME_REACHED_DIST = 0.5    # m: distância ao spawn p/ declarar "chegou em casa"
    LIFT_TICKS = 15            # ~1.5 s p/ o braço subir antes de dirigir
    DEPOSIT_TICKS = 20         # ~2.0 s parado após abrir a garra (deixa a bandeira assentar)
    REVERSE_SPEED = 0.10       # m/s: ré lenta p/ afastar a garra da bandeira depositada
    REVERSE_DIST = 0.5         # m: distância de ré antes de fechar a garra
    REVERSE_MAX_TICKS = 80     # cap de segurança (~8 s) caso a pose não atualize
    CLOSE_SETTLE_TICKS = 10    # ~1.0 s p/ a garra fechar antes de encerrar

    # Servovisão do ajuste fino (ADJUSTING) + percepção da bandeira.
    KP_BEARING = 0.3       # ganho proporcional do erro horizontal (ADJUSTING)
    CENTER_TOL = 0.15      # |erro| abaixo disto = bandeira "centralizada"
    AREA_MIN = 10          # px²: blob menor = ruído/longe → ignora
    AREA_NEAR = 1500       # px²: blob ≥ isto = "chegou" (arrival VISUAL). TUNAR.
    FLAG_LABEL = 25        # blue_flag no labels_map (segmentação semântica)

    # REFINDING_FLAG — gira para reencontrar a bandeira perdida de vista.
    REFIND_ANGULAR = 0.5         # rad/s do giro de busca
    REFIND_TIMEOUT_TICKS = 120   # ~12 s (≈ uma volta) sem achar → desiste p/ EXPLORING

    # --- Busca DIRIGIDA (EXPLORING) ---
    # Navega a um WAYPOINT na zona azul (+x), onde a bandeira fixa fica em todos os mapas
    # (o robô spawna na zona vermelha em -x). É avistada a caminho (FOV ~90°) e dispara a
    # confirmação ANTES do waypoint → o waypoint é só uma DIREÇÃO, derivada do spawn
    # (agnóstico ao mapa). Se chegar sem avistá-la (oclusão), cai p/ REFINDING.
    SEARCH_FORWARD = 15.0        # m em +x do spawn → alvo ~x=+7 (centro da zona azul)
    SEARCH_REACHED_DIST = 1.0    # m: "chegou ao waypoint" (gatilho do fallback)

    # --- Planejamento GLOBAL (mapa de ocupação /grid_map + A*) [Trab. 2] ---
    # O desvio reativo trava em PAREDES (mínimo local); o A* enxerga o mapa todo e acha
    # o caminho pela porta. Desconhecido = LIVRE (otimismo) + replanejamento contínuo.
    INFLATION_M = 0.30           # m: infla obstáculos pelo raio do corpo (~0.16) + margem.
    # TETO ~0.33: o gargalo do arena_paredes é o corredor ~1.05 m (ponta da parede-U vs
    # muro). Folga ≈ 1.05 − 2·INFLATION_M; acima de ~0.35 o corredor fecha p/ o A*.
    REPLAN_PERIOD = 0.5          # s: replaneja ~2x/s contra o mapa que cresce
    # Seguidor (carrot/pure-pursuit sobre os cantos do A*): mira no próximo canto além de
    # WAYPOINT_REACHED. Segmentos entre cantos são livres → dirigir reto até ele é seguro.
    WAYPOINT_REACHED = 0.3       # m: dentro disto o canto conta como "passado"

    # --- Navegação por META no frame odom (go-to-point) ---
    # Ao confirmar a bandeira, congela um PONTO no frame odom e navega até ele. Perder o
    # pixel (ex.: num desvio) não nos faz parar — a meta fica lembrada.
    CAM_HFOV = 1.57              # rad: hfov da câmera de segmentação (URDF)
    DEFAULT_FLAG_RANGE = 2.5     # m: alcance assumido p/ projetar a meta quando o LIDAR não
                                 # vê a bandeira (>range_max). Só dá DIREÇÃO; refina perto.
    GOAL_KP_ANG = 1.2           # rad/s por rad de erro de bearing
    GOAL_KP_LIN = 0.5           # m/s por m de erro de distância
    GOAL_V_MAX = 0.35           # m/s máx à frente
    GOAL_W_MAX = 1.0            # rad/s máx de giro
    GOAL_ALIGN_TOL = 0.6        # rad: |bearing| acima disto → só gira, não avança
    # Anti-deadlock: a chegada é VISUAL (AREA_NEAR); se perdemos o pixel mas alcançamos a
    # meta lembrada, vai REFINDING (gira p/ reaver) em vez de estacionar parado.
    GOAL_REACHED_DIST = 0.5     # m: "alcançou a meta lembrada" (sem a bandeira à vista)

    # --- Guarda de PROXIMIDADE LATERAL (camada reativa em _drive_to_point) ---
    # Projeta cada feixe no frame do robô: fwd = r·cosθ, lat = r·sinθ. Dos feixes na faixa
    # do corpo (fwd ∈ [LAT_BACK, LAT_FRONT]), empurra p/ longe da parede mais próxima ao
    # lado + freia, sem trocar de estado. Projeção CARTESIANA (não faixas de ângulo fixas)
    # = robusta a PAREDES FINAS em ângulo rasante (todos os feixes dão ~a mesma |lat|).
    LAT_BACK = -0.20            # m: limite traseiro da faixa (ignora o que está atrás do eixo)
    LAT_FRONT = 0.55           # m: limite dianteiro (rodas/flanco + ponta do braço)
    # Folga exigida PROPOSITALMENTE < meia-largura do corredor (~0.52 m): centrado, ambas
    # as paredes ficam fora da folga → não trava simétrico; só corrige quando DERIVA.
    LAT_CLEAR = 0.40           # m: corpo ~0.16 + margem; parede mais perto → repulsão
    LAT_BAND = 0.18            # m: largura da rampa (clear→repulsão máx) por feixe
    SIDE_KP_ANG = 1.1           # rad/s de empurrão angular na repulsão máxima
    SIDE_BRAKE = 0.5            # fração máx de freio linear (não zera o avanço)

    # EMERGENCY_DIST — reflexo frontal ao SEGUIR o A* (subordinado ao plano). Hoje é código
    # morto (_should_avoid não dispara com caminho); mantido p/ uma rede futura.
    EMERGENCY_DIST = 0.45      # m (origem do LIDAR): braço até ~0.40 → 0.45 = "quase tocando"

    def _is_carrying(self) -> bool:
        # True quando o robô está carregando a bandeira (corpo ampliado).
        # Inclui OBSTACLE_AVOIDANCE chamado a partir de RETURNING_HOME (previous_state
        # registra quem entrou no desvio), senão a zona-morta do braço e os limiares
        # ampliados seriam desligados DURANTE o desvio — travando o robô.
        if self.current_state == self.States.RETURNING_HOME:
            return True
        if (self.current_state == self.States.OBSTACLE_AVOIDANCE
                and self.previous_state == self.States.RETURNING_HOME):
            return True
        return False

    def _carry_extra(self) -> float:
        # Extensão EXTRA do corpo quando carregando a bandeira. Soma-se aos limiares
        # de desvio (FRONT_BLOCK_DIST, DANGER_DIST, CLEAR_DIST, ROTATE_SAFE_DIST) para
        # que o robô mantenha mais folga com o corpo ampliado, SEM filtrar leituras do
        # LIDAR — assim não perde de vista obstáculos reais próximos.
        return self.CARRY_BODY_EXTRA if self._is_carrying() else 0.0

    def _carry_dead_zone(self, scan):
        # Conjunto de índices do LIDAR a IGNORAR quando carregando: ±CARRY_ARM_HALF_DEG ao
        # redor do índice 0 (frente exata), onde o ponto de montagem do braço levantado cria
        # uma leitura fantasma. Fora desse cone estreito (~5 feixes), o LIDAR lê normalmente.
        arm_k = int(round(math.radians(self.CARRY_ARM_HALF_DEG) / scan.angle_increment))
        n = len(scan.ranges)
        return set(range(0, arm_k + 1)) | set(range(n - arm_k, n))

    def _front_min_dist(self) -> float:
        # Menor distância válida no arco frontal (±FRONT_ARC_HALF). Fonte única de
        # verdade para "bloqueado" e "perigo". Retorna 0.0 quando cego (sem scan /
        # tudo NaN) → ambos os limiares disparam → comportamento seguro.
        scan = self.latest_scan
        if scan is None:
            return 0.0

        k = int(round(self.FRONT_ARC_HALF / scan.angle_increment))
        n = len(scan.ranges)
        # Frente = índice 0; janela embrulha: 0..k (à esquerda) e n-k..n-1 (à direita).
        idxs = list(range(0, k + 1)) + list(range(n - k, n))

        # Ao carregar, exclui a zona-morta do braço (±2.5° do centro).
        if self._is_carrying():
            dead = self._carry_dead_zone(scan)
            idxs = [i for i in idxs if i not in dead]

        # inf = nada detectado no alcance = LIVRE (mantém). Descarta só inválidos:
        # NaN e leituras abaixo do range_min (os 0.0 que aparecem no echo).
        valid = [scan.ranges[i] for i in idxs
                 if not math.isnan(scan.ranges[i]) and scan.ranges[i] >= scan.range_min]
        if not valid:
            return 0.0  # cego de verdade (tudo NaN) → perigo por segurança
        return min(valid)

    def _front_min_bearing(self):
        # Bearing (rad) do feixe de MENOR range no arco frontal (±FRONT_ARC_HALF) =
        # direção do objeto mais próximo à frente. Na captura (a <1 m do alvo) esse
        # objeto é o MASTRO → dá o alinhamento lateral correto independentemente da
        # máscara de segmentação (que pode incluir o painel deslocado). +=esq/CCW,
        # −=dir/CW. Retorna None se cego (sem scan / só inválidos).
        scan = self.latest_scan
        if scan is None:
            return None
        k = int(round(self.FRONT_ARC_HALF / scan.angle_increment))
        n = len(scan.ranges)
        idxs = list(range(0, k + 1)) + list(range(n - k, n))  # frente, embrulhando
        # Ao carregar, exclui a zona-morta do braço (±2.5° do centro).
        if self._is_carrying():
            dead = self._carry_dead_zone(scan)
            idxs = [i for i in idxs if i not in dead]
        best_r, best_i = float('inf'), None
        for i in idxs:
            r = scan.ranges[i]
            if math.isnan(r) or r < scan.range_min:
                continue
            if r < best_r:
                best_r, best_i = r, i
        if best_i is None:
            return None
        return self._norm_angle(best_i * scan.angle_increment)  # idx→ângulo, p/ (-π,π]

    def _front_blocked(self) -> bool:
        return self._front_min_dist() < self.FRONT_BLOCK_DIST + self._carry_extra()

    def _front_in_danger(self) -> bool:
        # Mais apertado que _front_blocked: algo perigosamente perto da frente.
        return self._front_min_dist() < self.DANGER_DIST + self._carry_extra()

    def _side_clearances(self):
        # Espaço livre médio nos setores laterais (~20°..100°) de cada lado.
        # Retorna (esquerda, direita). (LIDAR: índice cresce CCW; frente = índice 0.)
        scan = self.latest_scan
        if scan is None:
            return 0.0, 0.0
        inc = scan.angle_increment
        n = len(scan.ranges)
        lo = int(round(math.radians(20.0) / inc))
        hi = int(round(math.radians(100.0) / inc))

        def sector_clearance(idxs):
            vals = []
            for i in idxs:
                r = scan.ranges[i % n]
                if math.isnan(r) or r < scan.range_min:
                    continue
                vals.append(min(r, scan.range_max))  # inf/longe → range_max (livre)
            return sum(vals) / len(vals) if vals else 0.0

        left = sector_clearance(range(lo, hi + 1))           # CCW da frente = esquerda
        right = sector_clearance(range(n - hi, n - lo + 1))  # CW da frente = direita
        return left, right

    def _side_repulsion(self):
        # Repulsão lateral por PROJEÇÃO CARTESIANA (robusta a paredes finas). Para cada feixe
        # válido: fwd = r·cosθ, lat = r·sinθ. Mantém só os que estão na faixa do corpo
        # (LAT_BACK ≤ fwd ≤ LAT_FRONT). Se |lat| < LAT_CLEAR, gera força ∈ (0,1] que cresce
        # até 1 dentro de LAT_BAND conforme a parede se aproxima. Retorna a MAIOR força de cada
        # lado: (s_esq, s_dir). Carregando a bandeira, exige mais folga (corpo ampliado).
        scan = self.latest_scan
        if scan is None:
            return 0.0, 0.0
        inc = scan.angle_increment
        n = len(scan.ranges)
        clear = self.LAT_CLEAR + self._carry_extra()
        s_left = s_right = 0.0
        for i in range(n):
            r = scan.ranges[i]
            if math.isnan(r) or math.isinf(r) or r < scan.range_min or r > scan.range_max:
                continue                       # inf/inválido/fora do alcance = livre
            theta = self._norm_angle(i * inc)  # ângulo do feixe em (-π, π]
            fwd = r * math.cos(theta)
            if fwd < self.LAT_BACK or fwd > self.LAT_FRONT:
                continue                       # fora da faixa lateral do corpo
            lat = r * math.sin(theta)          # + = esquerda (CCW), − = direita (CW)
            d = abs(lat)
            if d >= clear:
                continue
            strength = min(1.0, (clear - d) / self.LAT_BAND)   # rampa: 0→1 ao aproximar
            if lat >= 0.0:
                s_left = max(s_left, strength)
            else:
                s_right = max(s_right, strength)
        return s_left, s_right

    def _freest_turn_dir(self) -> float:
        # +1 = esquerda (CCW), -1 = direita (CW): vira para o lado MAIS LIVRE.
        left, right = self._side_clearances()
        return 1.0 if left >= right else -1.0

    def _flag_side(self):
        # Sentido em que a bandeira está: +1 (esquerda) / -1 (direita), ou None se sem
        # meta/pose ou praticamente à frente (aí o lado é indiferente → deixa o LIDAR).
        if self.flag_goal is None or self.pose is None:
            return None
        x, y, yaw = self.pose
        gx, gy = self.flag_goal
        berr = self._norm_angle(math.atan2(gy - y, gx - x) - yaw)
        if abs(berr) < 0.1:
            return None
        return 1.0 if berr > 0 else -1.0

    def _choose_turn_dir(self) -> float:
        # FOCO NA BANDEIRA: contorna o cilindro pelo lado em que a bandeira está, para
        # reencontrá-la ao rodear — em vez de virar para o lado mais livre, que pode ser
        # o OPOSTO à bandeira. Sem meta (explorando) → lado mais livre. Segurança: se o
        # lado da bandeira for bem mais bloqueado (parede), cai p/ o lado mais livre.
        # Carregando a bandeira (RETURNING_HOME): sem foco na bandeira, lado mais livre.
        if self._is_carrying():
            return self._freest_turn_dir()
        side = self._flag_side()
        if side is None:
            return self._freest_turn_dir()
        left, right = self._side_clearances()
        flag_clear = left if side > 0 else right
        other_clear = right if side > 0 else left
        if flag_clear < 0.6 * other_clear:
            return self._freest_turn_dir()
        return side

    def _enter_avoidance(self):
        # Ponto único de entrada no desvio: registra quem chamou (p/ voltar depois),
        # zera o relógio de compromisso e TRAVA o sentido do giro (lado mais livre).
        self.previous_state = self.current_state
        self.avoid_ticks = 0
        # Perto demais p/ girar com segurança (o braço varre 0.4 m)? Começa dando RÉ;
        # senão, já começa girando para o lado livre/da bandeira.
        if self._front_min_dist() < self.ROTATE_SAFE_DIST + self._carry_extra():
            self.avoid_phase = 'BACKUP'
        else:
            self.avoid_phase = 'TURN'
        self.avoid_turn_dir = self._choose_turn_dir()  # foco na bandeira (se houver meta)
        return self.States.OBSTACLE_AVOIDANCE

    # --- Navegação go-to-point (frame odom) ---

    @staticmethod
    def _norm_angle(a: float) -> float:
        # Normaliza um ângulo para (-π, π], evitando saltos de ±2π no erro de bearing.
        return math.atan2(math.sin(a), math.cos(a))

    def _dist_to(self, gx: float, gy: float) -> float:
        # Distância euclidiana da pose atual até (gx, gy). Exige self.pose != None.
        x, y, _ = self.pose
        return math.hypot(gx - x, gy - y)

    def _range_at_bearing(self, bearing: float) -> float:
        # Lê o LIDAR no ângulo 'bearing' → range. Se o feixe não retorna nada válido
        # (bandeira além de range_max), devolve DEFAULT_FLAG_RANGE — só dá DIREÇÃO; a
        # distância se refina sozinha quando a bandeira entra no alcance do LIDAR.
        scan = self.latest_scan
        if scan is None:
            return self.DEFAULT_FLAG_RANGE
        n = len(scan.ranges)
        idx = int(round(bearing / scan.angle_increment)) % n
        r = scan.ranges[idx]
        if math.isnan(r) or math.isinf(r) or r < scan.range_min or r > scan.range_max:
            return self.DEFAULT_FLAG_RANGE
        return r

    def _update_flag_goal(self, cx: float):
        # Projeta o centroide da bandeira no frame odom → ponto-meta (a MEMÓRIA).
        # Sem pose não há frame fixo onde ancorar; aborta silenciosamente.
        if self.pose is None or self.image_width is None:
            return
        x, y, yaw = self.pose
        err = self._flag_bearing_error(cx)        # + = bandeira à direita
        bearing = -err * (self.CAM_HFOV / 2.0)    # à direita = bearing negativo (CW)
        rng = self._range_at_bearing(bearing)
        self.flag_goal = (x + rng * math.cos(yaw + bearing),
                          y + rng * math.sin(yaw + bearing))

    def _drive_to_point(self, gx: float, gy: float) -> float:
        # Controlador uniciclo: gira para o alvo e avança proporcional à distância,
        # freando quando desalinhado. Retorna a distância restante (inf se sem pose).
        if self.pose is None:
            self._publish_cmd(0.0, 0.0)
            return float('inf')
        x, y, yaw = self.pose
        dist = math.hypot(gx - x, gy - y)
        berr = self._norm_angle(math.atan2(gy - y, gx - x) - yaw)
        ang = self.GOAL_KP_ANG * berr
        lin = self.GOAL_KP_LIN * dist
        lin *= max(0.0, 1.0 - abs(berr) / self.GOAL_ALIGN_TOL)  # só avança alinhado

        # --- Camada reativa: repulsão LATERAL (campo potencial, footprint inflado) ---
        # O cone frontal (±25°) não vê cilindros rente ao flanco/roda. Aqui empurramos
        # para LONGE do lado próximo e freamos, proporcional à força (max dos feixes).
        # Não troca de estado: a meta lembrada (flag_goal) segue valendo.
        s_left, s_right = self._side_repulsion()
        ang += self.SIDE_KP_ANG * (s_right - s_left)   # +ang=CCW: foge da esquerda → −
        lin *= 1.0 - self.SIDE_BRAKE * max(s_left, s_right)

        w_max = self.CARRY_W_MAX if self._is_carrying() else self.GOAL_W_MAX
        v_max = self.CARRY_V_MAX if self._is_carrying() else self.GOAL_V_MAX
        ang = max(-w_max, min(w_max, ang))
        lin = max(0.0, min(v_max, lin))
        self._publish_cmd(lin, ang)
        return dist

    def _has_path(self) -> bool:
        # Há um caminho A* utilizável (>=2 vértices) p/ seguir? Quando não há (planejador
        # ainda sem mapa/pose, ou A* não achou rota), o seguidor cai para o go-to-point reto.
        return len(self.path) >= 2

    def _follow_path(self, gx: float, gy: float) -> float:
        # SEGUIDOR DE CAMINHO (A*): em vez de mirar reto na meta final (que pode estar atrás
        # de uma parede), mira no PRÓXIMO canto do caminho global. Como _replan() recalcula o
        # caminho a partir da pose ATUAL ~2x/s, o caminho começa sempre junto ao robô; o
        # "carrot" é o 1º vértice (a partir do mais próximo) além de WAYPOINT_REACHED. Reusa
        # _drive_to_point (mesma camada reativa de repulsão lateral). Sem caminho → fallback
        # go-to-point reto (comportamento antigo, sem regressão). Retorna a distância à META
        # FINAL (não ao canto), p/ a lógica de chegada dos estados continuar igual.
        if self.pose is None or not self._has_path():
            return self._drive_to_point(gx, gy)
        x, y, _ = self.pose
        dists = [math.hypot(wx - x, wy - y) for (wx, wy) in self.path]
        c = min(range(len(self.path)), key=lambda i: dists[i])   # vértice mais próximo = progresso
        carrot = self.path[-1]                                   # default: último ponto
        for i in range(c, len(self.path)):
            if dists[i] > self.WAYPOINT_REACHED:
                carrot = self.path[i]                            # 1º canto à frente além do limiar
                break
        self.last_carrot = carrot   # guarda p/ o gate direcional de _should_avoid (carrot ahead?)
        self.get_logger().info(  # TEMP DIAG
            f"FOLLOW pose=({x:.2f},{y:.2f}) c={c}/{len(self.path)} "
            f"carrot=({carrot[0]:.2f},{carrot[1]:.2f}) dcarrot={math.hypot(carrot[0]-x,carrot[1]-y):.2f} "
            f"front={self._front_min_dist():.2f} pend=({self.path[-1][0]:.2f},{self.path[-1][1]:.2f})",
            throttle_duration_sec=1.0)
        self._drive_to_point(*carrot)
        return math.hypot(gx - x, gy - y)

    def _should_avoid(self) -> bool:
        # Gatilho do desvio reativo (SEGURANÇA), SUBORDINADO ao A*: com caminho válido NÃO
        # desvia — o A* já roteia pelas portas e a guarda LATERAL cuida do raspão sem trocar
        # de estado. Num corredor o caminho passa legitimamente a ~0.45 m do canto, então um
        # reflexo frontal giraria o robô p/ fora e entraria em LIVELOCK (o motivo de existir
        # o A*). Só no fallback SEM caminho vale a regra reativa antiga (perigo OU bloqueio).
        if self._has_path():
            return False
        return self._front_in_danger() or self._front_blocked()

    def _carrot_ahead(self) -> bool:
        # O carrot (próximo canto do A*) está grosso modo À FRENTE? Se está de lado, estamos
        # virando e a parede frontal não bloqueia o trajeto pretendido → não é emergência.
        if self.last_carrot is None or self.pose is None:
            return True
        x, y, yaw = self.pose
        cx, cy = self.last_carrot
        berr = abs(self._norm_angle(math.atan2(cy - y, cx - x) - yaw))
        return berr < self.GOAL_ALIGN_TOL   # mesma tolerância do go-to-point (~34°)

    def _flag_bearing_error(self, cx: float) -> float:
        # Erro horizontal normalizado do centroide: 0 = centro, +1 = borda direita,
        # -1 = borda esquerda. Fonte única da conversão pixel→bearing (usado pelo
        # ADJUSTING, pelo _flag_centered e pela projeção da meta em _update_flag_goal).
        half = self.image_width / 2.0
        return (cx - half) / half

    def _flag_centered(self) -> bool:
        centroid = self.latest_flag_centroid
        if centroid is None or self.image_width is None:
            return False
        cx, _ = centroid
        return abs(self._flag_bearing_error(cx)) < self.CENTER_TOL

    # --- Transições (Fase 1): retornam o próximo State, ou None para "permanecer" ---

    def transition_idle(self):
        # Espera o primeiro LaserScan antes de explorar: sair de IDLE sem dados
        # de scan = dirigir cego, sem nada para o desvio de obstáculo ler.
        if self.latest_scan is not None:
            return self.States.EXPLORING
        return None

    def transition_exploring(self):
        # Prioridade subsumption: SEGURANÇA antes do OBJETIVO. Com A* ativo, o desvio
        # reativo é REDE: dispara em perigo iminente sempre, e no bloqueio largo só sem
        # caminho (fallback). Seguindo o caminho, o A* roteia pelas portas.
        if self._should_avoid():
            return self._enter_avoidance()  # trava sentido + relógio de compromisso
        # 2) Bandeira visível? Vai confirmar.
        if self.latest_flag_centroid is not None:
            self.flag_seen_consecutive = 0  # zera o contador ao ENTRAR na confirmação
            return self.States.FLAG_FOUND_CONFIRMATION
        # 3) Chegou ao waypoint de busca SEM avistar a bandeira (oclusão/raro): gira no
        #    lugar p/ varrer a zona (REFINDING), em vez de estacionar parado no waypoint.
        if (self.pose is not None and self.search_goal is not None
                and self._dist_to(*self.search_goal) < self.SEARCH_REACHED_DIST):
            self.refind_ticks = 0
            return self.States.REFINDING_FLAG
        # 4) Nada de interessante: segue a busca dirigida rumo ao waypoint.
        return None

    def transition_obstacle_avoidance(self):
        self.avoid_ticks += 1

        if self.avoid_phase == 'BACKUP':
            # Dá ré até ter espaço p/ girar sem o braço raspar (ou estoura o teto de ré).
            if (self._front_min_dist() >= self.ROTATE_SAFE_DIST + self._carry_extra()
                    or self.avoid_ticks >= self.BACKUP_MAX_TICKS):
                self.avoid_phase = 'TURN'
                self.avoid_ticks = 0
                self.avoid_turn_dir = self._choose_turn_dir()  # foco na bandeira
            return None  # ainda dando ré

        if self.avoid_phase == 'TURN':
            # COMPROMISSO: gira pelo menos AVOID_MIN_TICKS (mata chatter, ~30° real).
            if self.avoid_ticks < self.AVOID_MIN_TICKS:
                return None
            # HISTERESE: só "libera" com folga LARGA (CLEAR_DIST > FRONT_BLOCK_DIST).
            # Se girou demais sem achar (beco/canto), RELAXA p/ FRONT_BLOCK_DIST —
            # anti-livelock (não gira para sempre → não atravessa a parede).
            extra = self._carry_extra()
            required = self.CLEAR_DIST + extra
            if self.avoid_ticks >= self.AVOID_MAX_TICKS:
                required = self.FRONT_BLOCK_DIST + extra
            if self._front_min_dist() >= required:
                self.avoid_phase = 'ESCAPE'   # frente livre → ESCAPA p/ frente (lateral)
                self.escape_ticks = 0
            return None  # segue no desvio (girando ou já trocou p/ ESCAPE)

        # Fase ESCAPE: anda para frente na direção livre por ESCAPE_TICKS, deslocando-se
        # ao redor do obstáculo. Se algo voltar a ficar PERIGOSO à frente, NÃO gira no
        # lugar (o braço raspa) — DÁ RÉ primeiro p/ abrir espaço, depois gira.
        if self._front_in_danger():
            self.avoid_phase = 'BACKUP'
            self.avoid_ticks = 0
            return None
        self.escape_ticks += 1
        if self.escape_ticks >= self.ESCAPE_TICKS:
            # Contornou o suficiente: volta a navegar para a meta lembrada.
            if (self.previous_state == self.States.EXPLORING
                    and self.latest_flag_centroid is not None):
                self.flag_seen_consecutive = 0
                return self.States.FLAG_FOUND_CONFIRMATION
            return self.previous_state
        return None  # ainda escapando para frente

    def transition_flag_found_confirmation(self):
        # Falha: bandeira sumiu antes de confirmar → era falso positivo → explora.
        if self.latest_flag_centroid is None:
            return self.States.EXPLORING
        # Sucesso: vista por ticks consecutivos suficientes (~2 s) → vai até ela.
        self.flag_seen_consecutive += 1
        if self.flag_seen_consecutive >= self.CONFIRM_TICKS:
            return self.States.GOING_TO_FLAG
        return None  # ainda confirmando: fica parada olhando

    def transition_going_to_flag(self):
        # 1) Sem meta estimada (não deveria pós-confirmação)? Procura visualmente.
        #    NOTE: perder o PIXEL não cai mais aqui — navegamos para a meta lembrada.
        if self.flag_goal is None:
            self.refind_ticks = 0
            return self.States.REFINDING_FLAG
        # 2) Chegou? Arrival VISUAL: o blob da bandeira ficou grande o bastante (perto).
        #    A bandeira pode não estar no plano do LIDAR, então a ÁREA é o único sinal
        #    confiável de proximidade. Vence o desvio (chegada antes de bloqueio).
        if (self.latest_flag_centroid is not None
                and self.latest_flag_area is not None
                and self.latest_flag_area >= self.AREA_NEAR):
            return self.States.ADJUSTING_POSITION_TO_COLLECT_FLAG
        # 3) Obstáculo à frente? Desvia (override, rede de segurança subordinada ao A*).
        #    Ao voltar, a meta segue lembrada.
        if self._should_avoid():
            return self._enter_avoidance()
        # 4) Anti-deadlock: alcançamos a meta lembrada mas SEM a bandeira à vista
        #    (perdemos o pixel perto do ponto). Não fica estacionado: gira p/ reaver.
        if (self.latest_flag_centroid is None and self.pose is not None
                and self._dist_to(*self.flag_goal) < self.GOAL_REACHED_DIST):
            self.refind_ticks = 0
            return self.States.REFINDING_FLAG
        # 5) Segue navegando para a meta.
        return None

    def transition_refinding_flag(self):
        # Reachou a bandeira? Volta a persegui-la.
        if self.latest_flag_centroid is not None:
            return self.States.GOING_TO_FLAG
        # Girou uma volta inteira sem achar? Desiste e volta a explorar.
        self.refind_ticks += 1
        if self.refind_ticks >= self.REFIND_TIMEOUT_TICKS:
            return self.States.EXPLORING
        return None  # continua girando à procura

    def transition_adjusting_position(self):
        # 1) Perdeu a bandeira durante o ajuste fino? Procura de novo (igual ao
        #    GOING_TO_FLAG) em vez de girar às cegas para um centroide inexistente.
        if self.latest_flag_centroid is None:
            self.refind_ticks = 0
            return self.States.REFINDING_FLAG
        # 2) Centralizada dentro da tolerância? Orientação ajustada → coleta.
        if self._flag_centered():
            self.collect_ticks = 0          # zera o relógio da sub-fase ao ENTRAR
            self.collect_phase = 'OPEN'     # captura começa abrindo a garra
            return self.States.COLLECTING_FLAG
        # 3) Ainda torta: continua girando para centralizar.
        return None

    def transition_collecting_flag(self):
        # Máquina de sub-fases da captura: OPEN → CREEP → CLOSE → RETURNING_HOME.
        self.collect_ticks += 1

        if self.collect_phase == 'OPEN':
            # Rede de segurança: perder a bandeira AQUI (ainda não comprometido) → reprocura.
            if self.latest_flag_centroid is None:
                self.refind_ticks = 0
                return self.States.REFINDING_FLAG
            if self.collect_ticks >= self.OPEN_TICKS:   # garra aberta e assentada
                self.collect_phase = 'CREEP'
                self.collect_ticks = 0
            return None

        if self.collect_phase == 'CREEP':
            # Ainda não comprometido: perder a bandeira durante o avanço → reprocura.
            if self.latest_flag_centroid is None:
                self.refind_ticks = 0
                return self.States.REFINDING_FLAG
            # Mastro na distância de preensão (LIDAR) OU teto de creep → FECHA.
            if (self._front_min_dist() <= self.GRASP_DIST
                    or self.collect_ticks >= self.CREEP_MAX_TICKS):
                self.collect_phase = 'CLOSE'
                self.collect_ticks = 0
            return None

        # CLOSE: COMPROMETIDO — não reprocura mesmo se a garra ocultar a bandeira.
        if self.collect_ticks >= self.CLOSE_TICKS:
            # Confirmação SEM sensor de contato: a bandeira ainda visível (mastro acima
            # da garra) sugere captura; sumida sugere que foi derrubada. Sinal fraco,
            # registrado p/ inspeção no teste (a reforçar depois, ex. teste de empurrão).
            confirmed = self.latest_flag_centroid is not None
            self.get_logger().info(
                "BANDEIRA CAPTURADA (mastro visível). -> RETURNING_HOME" if confirmed
                else "Garra fechada mas mastro SUMIU (captura incerta). -> RETURNING_HOME")
            self.return_phase = 'LIFT'      # inicia o retorno levantando o braço
            self.return_ticks = 0
            return self.States.RETURNING_HOME
        return None

    def transition_returning_home(self):
        # Sub-fases: LIFT (levanta braço) → NAVIGATE (dirige p/ home) → DEPOSIT (solta a
        # bandeira) → DONE (terminal: parado). Antes voltava p/ IDLE, mas o IDLE auto-reinicia
        # o EXPLORING → o robô re-explorava carregando a bandeira ("enlouquecia"). DONE é
        # TERMINAL (transição sempre None) → a missão de fato encerra ao chegar e depositar.
        self.return_ticks += 1

        if self.return_phase == 'LIFT':
            if self.return_ticks >= self.LIFT_TICKS:
                self.return_phase = 'NAVIGATE'
                self.return_ticks = 0
            return None  # ainda levantando o braço

        if self.return_phase == 'DEPOSIT':
            if self.return_ticks >= self.DEPOSIT_TICKS:   # garra aberta + 2 s parado
                self.get_logger().info("BANDEIRA DEPOSITADA na base — recuando 0.5 m de ré.")
                self.reverse_start = ((self.pose[0], self.pose[1])
                                      if self.pose is not None else None)
                self.return_phase = 'REVERSING'
                self.return_ticks = 0
            return None  # ainda soltando a bandeira (parado)

        if self.return_phase == 'REVERSING':
            backed = (self.reverse_start is not None and self.pose is not None
                      and self._dist_to(*self.reverse_start) >= self.REVERSE_DIST)
            # Cap de segurança: aborta a ré se a pose travar (não recua indefinidamente).
            if backed or self.return_ticks >= self.REVERSE_MAX_TICKS:
                self.return_phase = 'CLOSING'
                self.return_ticks = 0
            return None  # ainda dando ré

        if self.return_phase == 'CLOSING':
            if self.return_ticks >= self.CLOSE_SETTLE_TICKS:
                self.return_phase = 'DONE'
                self.return_ticks = 0
                self.get_logger().info("MISSÃO CONCLUÍDA — garra fechada, robô parado.")
            return None  # garra fechando

        if self.return_phase == 'DONE':
            return None  # TERMINAL: missão encerrada, robô parado (sem re-exploração)

        # Fase NAVIGATE: obstáculo à frente? Desvia (rede subordinada ao A*, igual GOING_TO_FLAG).
        if self._should_avoid():
            return self._enter_avoidance()

        # Chegou em casa? Inicia o depósito (NÃO volta p/ IDLE — quebra o loop de re-exploração).
        if (self.pose is not None and self.start_pose is not None
                and self._dist_to(self.start_pose[0], self.start_pose[1])
                < self.HOME_REACHED_DIST):
            self.get_logger().info("CHEGOU EM CASA — depositando a bandeira.")
            self.return_phase = 'DEPOSIT'
            self.return_ticks = 0
            return None

        return None  # segue navegando

    # =======================================================
    # Planejamento GLOBAL: mapa de ocupação (/grid_map) + A*
    # =======================================================

    def grid_callback(self, msg: OccupancyGrid):
        # Guarda o mapa de ocupação (numpy) + metadados. Inflação e A* rodam no
        # replanejamento (não aqui), p/ não pesar no callback assíncrono.
        info = msg.info
        self.map_info = (info.resolution, info.origin.position.x,
                         info.origin.position.y, info.width, info.height)
        self.occ_grid = np.array(msg.data, dtype=np.int8).reshape(info.height, info.width)

    def _world_to_cell(self, x, y):
        res, ox, oy, W, H = self.map_info
        return int((x - ox) / res), int((y - oy) / res)

    def _cell_to_world(self, gx, gy):
        res, ox, oy, W, H = self.map_info
        return (ox + (gx + 0.5) * res, oy + (gy + 0.5) * res)

    @staticmethod
    def _disk(radius_cells):
        # Elemento estruturante circular p/ a inflação (footprint do robô).
        r = radius_cells
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        return (x * x + y * y) <= r * r

    def _compute_blocked(self):
        # Células OCUPADAS (==100) infladas pelo raio do robô. Livre(0) e
        # DESCONHECIDO(-1) contam como livres (otimismo → avança e descobre).
        occ = (self.occ_grid == 100)
        res = self.map_info[0]
        rad = max(1, int(math.ceil(self.INFLATION_M / res)))
        return ndimage.binary_dilation(occ, structure=self._disk(rad))

    @staticmethod
    def _astar(blocked, start, goal):
        # A* 8-conexo, heurística euclidiana (admissível p/ 8 vizinhos → ótimo).
        # blocked: array (H,W) booleano. start/goal: (gx, gy). Devolve [(gx,gy),...]
        # do start ao goal, ou None se não há caminho.
        H, W = blocked.shape
        sx, sy = start
        gx, gy = goal

        def h(x, y):
            return math.hypot(x - gx, y - gy)

        SQ2 = math.sqrt(2.0)
        nbrs = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2))
        openh = [(h(sx, sy), 0.0, sx, sy)]
        came = {}
        gscore = {(sx, sy): 0.0}
        closed = set()
        while openh:
            f, g, x, y = heapq.heappop(openh)
            if (x, y) in closed:
                continue
            if (x, y) == (gx, gy):
                path = [(x, y)]
                while (x, y) in came:
                    x, y = came[(x, y)]
                    path.append((x, y))
                return path[::-1]
            closed.add((x, y))
            for dx, dy, cost in nbrs:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= W or ny < 0 or ny >= H or blocked[ny, nx]:
                    continue
                ng = g + cost
                if ng < gscore.get((nx, ny), float('inf')):
                    gscore[(nx, ny)] = ng
                    came[(nx, ny)] = (x, y)
                    heapq.heappush(openh, (ng + h(nx, ny), ng, nx, ny))
        return None

    @staticmethod
    def _nearest_free(blocked, cell, max_r=25):
        # Se a célula (robô ou meta) caiu DENTRO da inflação, acha a célula livre
        # mais próxima em anéis crescentes — senão o A* falharia por start/goal inválido.
        H, W = blocked.shape
        cx, cy = cell
        if 0 <= cx < W and 0 <= cy < H and not blocked[cy, cx]:
            return (cx, cy)
        for r in range(1, max_r + 1):
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    x, y = cx + dx, cy + dy
                    if 0 <= x < W and 0 <= y < H and not blocked[y, x]:
                        return (x, y)
            for dy in range(-r + 1, r):
                for dx in (-r, r):
                    x, y = cx + dx, cy + dy
                    if 0 <= x < W and 0 <= y < H and not blocked[y, x]:
                        return (x, y)
        return None

    def _simplify_path(self, cells):
        # Colapsa células colineares: mantém só os pontos onde a DIREÇÃO muda
        # (A* anda em passos unitários, então direção igual = mesmo segmento reto).
        if len(cells) < 3:
            return [self._cell_to_world(*c) for c in cells]
        out = [cells[0]]
        for i in range(1, len(cells) - 1):
            ax, ay = cells[i - 1]
            bx, by = cells[i]
            cx, cy = cells[i + 1]
            if (bx - ax, by - ay) != (cx - bx, cy - by):
                out.append(cells[i])
        out.append(cells[-1])
        return [self._cell_to_world(*c) for c in out]

    def _plan_goal(self):
        # Meta do planejador: durante o retorno → casa; senão → bandeira ou waypoint.
        if self.current_state == self.States.RETURNING_HOME and self.start_pose is not None:
            return (self.start_pose[0], self.start_pose[1])
        if self.flag_goal is not None:
            return self.flag_goal
        return self.search_goal

    def _replan(self):
        # Recalcula o caminho global robô→meta e publica /plan (RViz). Guarda
        # self.path (lista de (x,y) no mundo) p/ o seguidor de caminho (Etapa 3).
        if self.occ_grid is None or self.pose is None:
            self.get_logger().warn(  # TEMP DIAG
                f"REPLAN bail: occ_grid={self.occ_grid is not None} pose={self.pose is not None}",
                throttle_duration_sec=1.0)
            return
        goal = self._plan_goal()
        if goal is None:
            self.get_logger().warn("REPLAN bail: goal=None", throttle_duration_sec=1.0)  # TEMP DIAG
            return
        blocked = self._compute_blocked()
        start = self._nearest_free(blocked, self._world_to_cell(self.pose[0], self.pose[1]))
        if start is not None:
            blocked[start[1], start[0]] = False  # nunca bloqueia a célula do próprio robô
        gcell = self._nearest_free(blocked, self._world_to_cell(goal[0], goal[1]))
        if start is None or gcell is None:
            self.path = []
            self._publish_plan([])
            self.get_logger().warn(  # TEMP DIAG
                f"REPLAN no cell: start={start} gcell={gcell} "
                f"goalworld=({goal[0]:.1f},{goal[1]:.1f}) occ100={(self.occ_grid==100).sum()}",
                throttle_duration_sec=1.0)
            return
        cells = self._astar(blocked, start, gcell)
        self.path = self._simplify_path(cells) if cells else []
        self._publish_plan(self.path)
        self.get_logger().info(  # TEMP DIAG
            f"REPLAN ok: start={start} gcell={gcell} cells={len(cells) if cells else 0} "
            f"wpts={len(self.path)} occ100={(self.occ_grid==100).sum()}",
            throttle_duration_sec=1.0)

    def _maybe_replan(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_replan < self.REPLAN_PERIOD:
            return
        self.last_replan = now
        self._replan()

    def _publish_plan(self, waypoints):
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()
        for (wx, wy) in waypoints:
            ps = PoseStamped()
            ps.header.frame_id = "map"
            ps.pose.position.x = float(wx)
            ps.pose.position.y = float(wy)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.plan_pub.publish(path)

    def __init__(self):
        super().__init__('mission_fsm')

        self.behaviors = {
            self.States.IDLE: self.behavior_idle,
            self.States.EXPLORING: self.behavior_exploring,
            self.States.OBSTACLE_AVOIDANCE: self.behavior_obstacle_avoidance,
            self.States.FLAG_FOUND_CONFIRMATION: self.behavior_flag_found_confirmation,
            self.States.GOING_TO_FLAG: self.behavior_going_to_flag,
            self.States.REFINDING_FLAG: self.behavior_refinding_flag,
            self.States.ADJUSTING_POSITION_TO_COLLECT_FLAG: self.behavior_adjusting_position,
            self.States.COLLECTING_FLAG: self.behavior_collecting_flag,
            self.States.RETURNING_HOME: self.behavior_returning_home
        }

        # Transições por estado (espelha self.behaviors). Estados sem aresta
        # ainda não entram aqui; tick() usa .get() e trata ausência como "permanecer".
        self.transitions = {
            self.States.IDLE: self.transition_idle,
            self.States.EXPLORING: self.transition_exploring,
            self.States.OBSTACLE_AVOIDANCE: self.transition_obstacle_avoidance,
            self.States.FLAG_FOUND_CONFIRMATION: self.transition_flag_found_confirmation,
            self.States.GOING_TO_FLAG: self.transition_going_to_flag,
            self.States.REFINDING_FLAG: self.transition_refinding_flag,
            self.States.ADJUSTING_POSITION_TO_COLLECT_FLAG: self.transition_adjusting_position,
            self.States.COLLECTING_FLAG: self.transition_collecting_flag,
            self.States.RETURNING_HOME: self.transition_returning_home,
        }

        # Estado inicial
        self.current_state = MissionFSM.States.IDLE
        # Registro escondido da subsumption: quem chamou OBSTACLE_AVOIDANCE,
        # para onde retornar quando a frente liberar.
        self.previous_state = None
        self.latest_scan = None
        self.latest_flag_centroid = None
        self.latest_flag_area = None        # área do maior blob (px²), proxy de distância
        self.image_width = None
        self.flag_seen_consecutive = 0
        self.last_flag_err = 0.0            # último erro de bearing visto (lado da bandeira)
        self.refind_ticks = 0               # ticks girando em REFINDING sem reencontrar
        self.collect_ticks = 0              # ticks na sub-fase atual da captura
        self.collect_phase = 'OPEN'         # sub-fase da captura: OPEN → CREEP → CLOSE
        self.avoid_ticks = 0                # ticks no desvio atual (compromisso mínimo)
        self.avoid_turn_dir = 1.0           # sentido travado do giro de desvio (+esq/-dir)
        self.avoid_phase = 'TURN'           # fase do desvio: 'TURN' (gira) ou 'ESCAPE' (anda)
        self.escape_ticks = 0               # ticks andando na fase ESCAPE
        self.return_phase = 'LIFT'          # sub-fase do retorno: LIFT → NAVIGATE
        self.return_ticks = 0               # ticks na sub-fase atual do retorno
        self.reverse_start = None           # (x, y) onde a ré pós-depósito começou
        self.pose = None                    # (x, y, yaw) no frame odom_gt; None até 1ª msg
        self.flag_goal = None               # (x, y) estimado da bandeira no frame odom
        self.start_pose = None              # pose de spawn registrada (1ª msg de odom)
        self.search_goal = None             # (x, y) waypoint da busca dirigida (zona azul)
        # Planejamento global (Trab. 2): mapa de ocupação + caminho A*.
        self.occ_grid = None                # numpy (H,W) do /grid_map; None até 1ª msg
        self.map_info = None                # (res, ox, oy, W, H) do /grid_map
        self.path = []                      # caminho A* atual: lista de (x, y) no mundo
        self.last_carrot = None             # último carrot seguido (p/ gate direcional do desvio)
        self.last_replan = 0.0              # relógio do replanejamento (s)

        # Publisher para comando de velocidade
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/diff_drive_base_controller/cmd_vel', 10)
        # Publisher dos comandos da garra (posição das juntas): [elevação, dir, esq]
        self.gripper_pub = self.create_publisher(Float64MultiArray, '/gripper_controller/commands', 10)
        # Publisher do caminho planejado (A*) para visualização no RViz
        self.plan_pub = self.create_publisher(Path, '/plan', 10)

        # Subscribers
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(Odometry, '/odom_gt', self.odom_callback, 10)
        self.create_subscription(Image, '/robot_cam/labels_map', self.camera_callback, 10)
        self.create_subscription(OccupancyGrid, '/grid_map', self.grid_callback, 10)

        # Utilizado para converter imagens ROS -> OpenCV
        self.bridge = CvBridge()

        # Timer para enviar comandos continuamente
        self.timer = self.create_timer(0.1, self.tick)

    def scan_callback(self, msg: LaserScan):
        # Callback assíncrono: só guarda a última varredura. Toda a decisão
        # (bloqueio/perigo/repulsão) acontece no tick(), lendo self.latest_scan.
        self.latest_scan = msg

    def imu_callback(self, msg: Imu):
        # IMU está modelada e publicada, mas o FSM usa o yaw do /odom_gt
        # (ground-truth, sem drift). Mantida assinada para evidenciar o sensor;
        # no-op por enquanto (fusão roda+IMU seria trabalho de um robô real).
        pass

    def odom_callback(self, msg: Odometry):
        # Pose ground-truth (sem drift) no frame odom_gt. Guarda (x, y, yaw):
        # x,y → distância à meta; yaw → erro de bearing no controlador go-to-point.
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # quaternion → yaw (rotação em Z): yaw = atan2(2(wz+xy), 1-2(y²+z²)).
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        self.pose = (p.x, p.y, yaw)
        # Registra a pose de SPAWN na 1ª msg e congela o waypoint de busca dirigida:
        # alvo ~15 m em +x (zona azul), agnóstico ao mapa. (start_pose também servirá
        # de "casa" para o retorno à base no Trab. 2 — registrada aqui, não usada ainda.)
        if self.start_pose is None:
            self.start_pose = (p.x, p.y, yaw)
            self.search_goal = (p.x + self.SEARCH_FORWARD, p.y)
        self.get_logger().info(
            f"pose x={p.x:+.2f} y={p.y:+.2f} yaw={math.degrees(yaw):+.0f}",
            throttle_duration_sec=2.0)

    def camera_callback(self, msg: Image):
        # labels_map: cada pixel = id do label semântico (não é cor!). Lê cru.
        label_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # Pode chegar como mono (H,W) ou multicanal (H,W,C) com o label replicado;
        # reduz para um canal de forma robusta.
        if label_img.ndim == 3:
            label_img = label_img[:, :, 0]

        # Largura da imagem na primeira frame (para o erro de bearing depois)
        if self.image_width is None:
            self.image_width = label_img.shape[1]


        # Máscara binária só do label da bandeira-alvo.
        mask = (label_img == self.FLAG_LABEL).astype(np.uint8) * 255

        # Detecta contornos (blobs)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self.latest_flag_centroid = None
            self.latest_flag_area = None
            return

        # Maior blob = bandeira (robustez contra fragmentos/ruído)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        M = cv2.moments(largest)
        # Rejeita blobs minúsculos: bandeira longe = poucos px e pisca; comprometer-se
        # com 3 px gera detecção instável que mata o GOING_TO_FLAG no frame seguinte.
        if area < self.AREA_MIN or M['m00'] == 0:
            self.latest_flag_centroid = None
            self.latest_flag_area = None
            return

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        self.latest_flag_centroid = (cx, cy)
        self.latest_flag_area = area
        # Percepção → ESTIMATIVA no mundo: projeta a bandeira como um ponto no frame
        # odom (a memória que o go-to-point persegue). Sem blob válido, NÃO mexe em
        # flag_goal — perder o pixel mantém a meta lembrada.
        self._update_flag_goal(cx)

    def tick(self):
        # FASE 1 — checagem de transição: pode reatribuir self.current_state.
        # Avaliada todo tick; só "dispara" quando a condição é verdadeira.
        transition_fn = self.transitions.get(self.current_state)
        if transition_fn is not None:
            next_state = transition_fn()
            if next_state is not None and next_state != self.current_state:
                self.get_logger().info(
                    f"{self.current_state.name} -> {next_state.name}")
                self.current_state = next_state

        # Visibilidade do estado (throttle p/ não poluir): 1x por segundo.
        self.get_logger().info(
            f"state={self.current_state.name} centroid={self.latest_flag_centroid} "
            f"area={self.latest_flag_area} confirm={self.flag_seen_consecutive}",
            throttle_duration_sec=1.0)

        # FASE 2 — despacho de comportamento: roda o estado em que estamos agora.
        self.behaviors[self.current_state]()

        # Planejamento global: recalcula o caminho A* + publica /plan ~2x/s. O caminho é
        # CONSUMIDO pelos estados de navegação via _follow_path (seguidor de cantos/carrot).
        self._maybe_replan()

def main(args=None):
    rclpy.init(args=args)
    node = MissionFSM()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
