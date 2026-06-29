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
        # Comando à garra: Float64MultiArray [elevação, braço dir, braço esq] em metros.
        # O controlador (JointGroupPositionController) SEGURA a última pose, então
        # republicar o mesmo valor a cada tick é idempotente e robusto a perdas de msg.
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
            self._publish_cmd(0.3, 0.0)
            return
        self._drive_to_point(*self.search_goal)


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


    def behavior_flag_found_confirmation(self):
        self._publish_cmd(0.0, 0.0)


    def behavior_going_to_flag(self):
        # Navega para a META LEMBRADA da bandeira (frame odom), NÃO para o pixel atual.
        # Assim, perder a bandeira de vista (ex.: durante um desvio) não nos faz parar:
        # seguimos para o ponto memorizado; a câmera só refina a meta quando a vê.
        if self.flag_goal is None:
            self._publish_cmd(0.0, 0.0)   # sem meta ainda; a transição trata (REFINDING)
            return
        dist = self._drive_to_point(*self.flag_goal)
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
        # Voltar para a base é OPCIONAL (Trab. 2). A missão graduável termina na coleta,
        # então este é o estado FINAL: fica PARADO (missão concluída). Mantido no grafo
        # do FSM como destino do COLLECTING_FLAG.
        self._publish_cmd(0.0, 0.0)
        self.get_logger().info("Missao concluida. Robo parado.", throttle_duration_sec=5.0)


    # --- Percepção: helpers de LIDAR ---

    # LIDAR diagnosticado via `ros2 topic echo /scan --once`:
    # angle_min=0, angle_max≈2π, ~1°/raio, range=[0.12, 3.5]. Índice 0 = FRENTE.
    FRONT_ARC_HALF = math.radians(25.0)   # meia-largura do arco frontal (rad)
    # O range do LIDAR é medido a partir da origem do sensor (lidar_joint em x=0,
    # sobre o base_link). Mas a frente FÍSICA do robô é o braço/garra, que vai de
    # x=0.2 até x≈0.4 (gripper_extension xyz=0.2; gripper_pole comp.=0.2). Logo a
    # folga real na ponta do braço = r − 0.40. Para parar com ~0.15 m de margem na
    # ponta + reação, disparamos quando r ≤ ~0.40 (braço) + 0.15 + reação ≈ 0.65 m.
    FRONT_BLOCK_DIST = 0.7                # m (medido da origem do LIDAR, não da ponta)
    # Limiar de PERIGO (< FRONT_BLOCK_DIST): mais apertado que "bloqueado". Usado na
    # fase ESCAPE: se durante o escape algo volta a ficar perigosamente perto à frente,
    # NÃO gira no lugar (o braço raspa) — interrompe e volta para BACKUP/ré primeiro.
    # ~0.5 m do LIDAR = ~0.1 m da ponta do braço: iminente.
    DANGER_DIST = 0.5                     # m

    # OBSTACLE_AVOIDANCE — desvio COMPROMETIDO (anti-chatter), camada 2 (subsumption):
    AVOID_ANGULAR = 0.5                   # rad/s do giro (sentido escolhido na entrada)
    # Histerese (Schmitt): ENTRA o desvio em FRONT_BLOCK_DIST (0.7), mas só SAI quando
    # a frente está livre por uma margem MAIOR. Limiares iguais = vibração na fronteira.
    CLEAR_DIST = 1.0                      # m: frente precisa estar livre até aqui p/ sair
    # Compromisso mínimo: depois de entrar, gira pelo menos N ticks antes de cogitar
    # sair. A 10 Hz, 10 ticks ≈ 1 s ≈ 30° a 0.5 rad/s — vira de verdade, não 1 tick.
    AVOID_MIN_TICKS = 10
    # Escape anti-livelock: se já girou tanto (~uma varredura ampla) sem achar uma
    # folga LARGA (CLEAR_DIST), RELAXA a exigência para FRONT_BLOCK_DIST — aceita
    # qualquer saída não-perigosa em vez de girar para sempre (e atravessar a parede
    # do mapa). A 10 Hz, 60 ticks ≈ 6 s ≈ 170° a 0.5 rad/s.
    AVOID_MAX_TICKS = 60
    # Fase ESCAPE (anti-mínimo-local): depois de girar e LIBERAR a frente, anda PARA
    # FRENTE nessa direção livre, deslocando-se LATERALMENTE ao redor do obstáculo.
    # Sem isto, o giro-no-lugar não translada e o robô ORBITA o obstáculo sem contornar.
    ESCAPE_FWD = 0.35            # m/s à frente durante o escape (mais rápido → arco maior)
    ESCAPE_TICKS = 20            # ~1.5 s ≈ 0.37 m de deslocamento lateral por episódio
    # CURVA do escape: em vez de fugir RETO (e perder a bandeira de lado), arqueia de
    # volta para o lado do obstáculo (= -avoid_turn_dir) enquanto anda → CONTORNA o
    # cilindro em arco. Como o obstáculo está ENTRE robô e bandeira, curvar de volta a
    # ele ≈ curvar de volta à bandeira. Raio ≈ ESCAPE_FWD/ESCAPE_CURVE ≈ 0.6 m.
    ESCAPE_CURVE = 0.2           # rad/s de curva (gira mais devagar → arco MAIOR/mais aberto)
    # Fase BACKUP: girar NO LUGAR perto de um obstáculo faz o BRAÇO (0.4 m à frente)
    # varrer um círculo de 0.4 m e RASPAR no cilindro. Então, se a frente está perto
    # demais para girar com segurança, dá RÉ primeiro para abrir espaço ao braço.
    ROTATE_SAFE_DIST = 0.6      # m: frente livre até aqui = seguro girar (raio braço ~0.4)
    AVOID_BACK = 0.15           # m/s de ré durante o BACKUP
    BACKUP_MAX_TICKS = 15       # teto de ré (~0.22 m) p/ não dar ré pra sempre

    # Confirmação da bandeira: nº de ticks consecutivos com a bandeira visível
    # antes de aceitar. A 10 Hz, 20 ticks ≈ 2 s. Contamos TICKS (taxa fixa), não
    # frames da câmera — por isso isto mede tempo de verdade. Se o tick virasse
    # variável, o robusto seria medir com get_clock().now().
    CONFIRM_TICKS = 20

    # COLLECTING_FLAG — captura REAL (garra) em SUB-FASES dentro do estado, espelhando
    # o padrão do OBSTACLE_AVOIDANCE: OPEN (abre a garra parado) → CREEP (avança devagar
    # centralizando até o mastro entrar entre as pinças) → CLOSE (fecha e aperta). A garra
    # SEGURA a última pose, então republicamos o comando a cada tick (idempotente).
    GRIPPER_OPEN = [0.0, -0.06, 0.06]   # [elevação, braço dir, braço esq]: pinças abertas ±6 cm
    GRIPPER_CLOSED = [0.0, 0.0, 0.0]    # pinças fechadas (fenda ~2 cm) → aperta o mastro de 6 cm
    OPEN_TICKS = 10        # ~1 s parado p/ as pinças abrirem antes de avançar
    CREEP_FWD = 0.08       # m/s: avanço lento e controlado na aproximação final
    # Distância de PREENSÃO (via LIDAR): o mastro (~0.4 m de altura) CRUZA o plano do
    # LIDAR (z=0.12); a garra aberta em ext=0 fica ABAIXO desse plano, então _front_min_dist
    # lê o MASTRO limpo. Para o mastro entre as pinças (pontas em x≈0.43, LIDAR em x=0) a
    # superfície dele fica a ~0.40 m. TUNAR vendo o front_min no log durante o teste.
    GRASP_DIST = 0.40      # m: para de avançar e FECHA quando o mastro chega a esta distância
    CREEP_MAX_TICKS = 120  # teto de SEGURANÇA (~0.95 m a 0.08 m/s); o LIDAR (GRASP_DIST)
                           # deve encerrar o creep ANTES — o teto só evita avançar sem fim
    CLOSE_TICKS = 15       # ~1.5 s parado p/ o aperto assentar antes de declarar a captura
    CREEP_KP_ANG = 1.5     # rad/s por rad de bearing do mastro (LIDAR) → centra no mastro

    # Servovisão do ajuste fino (ADJUSTING) + percepção da bandeira.
    KP_BEARING = 0.3       # ganho proporcional do erro horizontal (ADJUSTING)
    CENTER_TOL = 0.15      # |erro| abaixo disto = bandeira "centralizada"
    AREA_MIN = 10          # px²: blob menor que isto = ruído/longe demais → ignora
    AREA_NEAR = 1500       # px²: blob ≥ isto = perto o bastante p/ "chegou". Arrival
                           # VISUAL (a bandeira pode não estar no plano do LIDAR). TUNAR.
    FLAG_LABEL = 25        # blue_flag no labels_map (segmentação semântica)

    # REFINDING_FLAG — gira para reencontrar a bandeira perdida de vista.
    REFIND_ANGULAR = 0.5         # rad/s do giro de busca
    REFIND_TIMEOUT_TICKS = 120   # ~12 s (≈ uma volta) sem achar → desiste p/ EXPLORING

    # --- Busca DIRIGIDA (EXPLORING) ---
    # Em vez de andar reto às cegas, navega para um WAYPOINT no frame odom dentro da
    # zona azul (+x), onde a bandeira FIXA fica em TODOS os mapas (zona vermelha sempre
    # em -x → robô spawna lá; blue flag sempre em +x, na linha de centro y≈0). A bandeira
    # é avistada de relance a caminho (FOV ~90°) e dispara FLAG_FOUND_CONFIRMATION ANTES
    # de chegar ao waypoint — então o waypoint é, na prática, só uma DIREÇÃO. Alvo
    # derivado da POSE DE SPAWN (+x), não de coordenada fixa → agnóstico ao mapa. Só se
    # CHEGAR ao waypoint sem avistar a bandeira (oclusão, ex. arena_paredes) cai p/ busca
    # giratória (REFINDING), evitando estacionar parado (penalidade "robô parado"). Reusa
    # o controlador go-to-point já validado no GOING_TO_FLAG (mesma camada de desvio).
    SEARCH_FORWARD = 15.0        # m em +x a partir do spawn → alvo ~x=+7 (centro da zona azul)
    SEARCH_REACHED_DIST = 1.0    # m: "chegou ao waypoint de busca" (gatilho do fallback)

    # --- Planejamento GLOBAL (mapa de ocupação /grid_map + A*) [Trab. 2] ---
    # O desvio reativo trava em PAREDES (mínimo local); o A* enxerga o mapa inteiro e
    # acha o caminho pela porta. Desconhecido = LIVRE (otimismo) + replanejamento → o
    # robô avança, descobre as paredes pelo LIDAR e desvia recalculando o caminho.
    INFLATION_M = 0.25           # m: infla obstáculos pelo raio do corpo (~0.16) + margem
    REPLAN_PERIOD = 0.5          # s: replaneja ~2x/s contra o mapa que vai crescendo

    # --- Navegação por META no frame odom (go-to-point) ---
    # Ideia: ao confirmar a bandeira, congela um PONTO no frame odom e navega até ele.
    # Perder o pixel (ex.: durante um desvio) não nos faz parar — a meta é lembrada.
    CAM_HFOV = 1.57              # rad: hfov da câmera de segmentação (do URDF)
    DEFAULT_FLAG_RANGE = 2.5     # m: alcance ASSUMIDO p/ projetar a meta quando o LIDAR
                                 # não vê a bandeira (>range_max). Só dá DIREÇÃO; refina perto.
    GOAL_KP_ANG = 1.2           # rad/s por rad de erro de bearing
    GOAL_KP_LIN = 0.5           # m/s por m de erro de distância
    GOAL_V_MAX = 0.35           # m/s máx à frente
    GOAL_W_MAX = 1.0            # rad/s máx de giro
    GOAL_ALIGN_TOL = 0.6        # rad: |bearing| acima disto → só gira, não avança
    # Anti-deadlock: chegada é VISUAL (AREA_NEAR). Mas se perdemos o pixel e ainda
    # assim alcançamos a meta lembrada, o robô estacionaria sem nunca "chegar". Quando
    # isso acontece, vai REFINDING (gira p/ reaver) em vez de travar parado.
    GOAL_REACHED_DIST = 0.5     # m: "alcançou a meta lembrada" (sem a bandeira à vista)

    # --- Guarda de PROXIMIDADE LATERAL (camada reativa em GOING_TO_FLAG) ---
    # O cone frontal (±25°) NÃO enxerga cilindros que passam rente ao FLANCO/braço —
    # por isso o robô às vezes raspa o lado durante a perseguição. Esta camada lê uma
    # faixa "near-front" de cada lado e, como um CAMPO POTENCIAL, empurra o comando
    # para LONGE do lado próximo + freia, SEM trocar de estado (a meta lembrada
    # flag_goal segue valendo, então perder o pixel um instante não custa nada).
    SIDE_NEAR_LO_DEG = 25.0     # início da faixa lateral (logo após o cone frontal)
    SIDE_NEAR_HI_DEG = 110.0    # fim da faixa: cobre rodas/flanco e um pouco atrás deles
    # FADE do limiar com o ÂNGULO (= footprint inflado): perto da frente exigimos MAIS
    # espaço (estamos transladando para lá); perto de 90° o feixe aponta para o FLANCO/
    # roda, onde o corpo já ocupa quase toda a folga → limiar CAI. clear(θ) interpola
    # linearmente de _FRONT (em LO) até _SIDE (em HI).
    SIDE_CLEAR_FRONT = 0.55     # m: folga exigida na borda dianteira da faixa (25°)
    SIDE_CLEAR_SIDE = 0.22      # m: folga exigida no flanco (~110°) ≈ meia-largura+margem
    SIDE_BAND = 0.20            # m: largura da rampa (clear→repulsão máx) em cada feixe
    SIDE_KP_ANG = 0.9           # rad/s de empurrão angular na repulsão máxima
    SIDE_BRAKE = 0.6            # fração máx. de freio linear na repulsão máxima

    def _front_min_dist(self) -> float:
        # Menor distância válida no arco frontal (±FRONT_ARC_HALF). Fonte única de
        # verdade para "bloqueado" e "perigo". Retorna 0.0 quando cego (sem scan /
        # tudo NaN) → ambos os limiares disparam → comportamento seguro.
        scan = self.latest_scan
        if scan is None:
            return 0.0

        k = int(round(self.FRONT_ARC_HALF / scan.angle_increment))
        # Frente = índice 0; janela embrulha: 0..k (à esquerda) e n-k..n-1 (à direita).
        window = list(scan.ranges[:k + 1]) + list(scan.ranges[-k:])

        # inf = nada detectado no alcance = LIVRE (mantém). Descarta só inválidos:
        # NaN e leituras abaixo do range_min (os 0.0 que aparecem no echo).
        valid = [r for r in window if not math.isnan(r) and r >= scan.range_min]
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
        return self._front_min_dist() < self.FRONT_BLOCK_DIST

    def _front_in_danger(self) -> bool:
        # Mais apertado que _front_blocked: algo perigosamente perto da frente.
        return self._front_min_dist() < self.DANGER_DIST

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
        # Repulsão lateral com folga DEPENDENTE DO ÂNGULO (footprint inflado). Para cada
        # feixe no arco [LO, HI] de cada lado: se o range cai abaixo do clear(θ) daquele
        # ângulo, gera força ∈ (0,1] que cresce até 1 dentro de SIDE_BAND. Retorna a MAIOR
        # força de cada lado: (s_esq, s_dir). Sem scan → sem repulsão.
        scan = self.latest_scan
        if scan is None:
            return 0.0, 0.0
        inc = scan.angle_increment
        n = len(scan.ranges)
        lo = math.radians(self.SIDE_NEAR_LO_DEG)
        hi = math.radians(self.SIDE_NEAR_HI_DEG)
        span_ang = hi - lo

        def strength_at(r: float, theta: float) -> float:
            if math.isnan(r) or math.isinf(r) or r < scan.range_min:
                return 0.0  # inf/inválido = livre
            t = (theta - lo) / span_ang                      # 0 na frente → 1 no flanco
            clear = self.SIDE_CLEAR_FRONT + t * (self.SIDE_CLEAR_SIDE - self.SIDE_CLEAR_FRONT)
            if r >= clear:
                return 0.0
            return min(1.0, (clear - r) / self.SIDE_BAND)    # rampa: 0→1 conforme aproxima

        i_lo = int(round(lo / inc))
        i_hi = int(round(hi / inc))
        s_left = s_right = 0.0
        for i in range(i_lo, i_hi + 1):
            theta = i * inc
            s_left = max(s_left, strength_at(scan.ranges[i % n], theta))      # CCW = esq
            s_right = max(s_right, strength_at(scan.ranges[(-i) % n], theta))  # CW = dir
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
        if self._front_min_dist() < self.ROTATE_SAFE_DIST:
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

        ang = max(-self.GOAL_W_MAX, min(self.GOAL_W_MAX, ang))
        lin = max(0.0, min(self.GOAL_V_MAX, lin))
        self._publish_cmd(lin, ang)
        return dist

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
        # Prioridade subsumption: SEGURANÇA antes do OBJETIVO.
        # 1) Frente bloqueada? Desvia primeiro, mesmo que a bandeira esteja à vista.
        if self._front_blocked():
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
            if (self._front_min_dist() >= self.ROTATE_SAFE_DIST
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
            required = self.CLEAR_DIST
            if self.avoid_ticks >= self.AVOID_MAX_TICKS:
                required = self.FRONT_BLOCK_DIST
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
        # 3) Obstáculo à frente? Desvia (override). Ao voltar, a meta segue lembrada.
        if self._front_blocked():
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
            return self.States.RETURNING_HOME
        return None

    # RETURNING_HOME não tem transição: é estado FINAL (missão concluída, robô parado).

    # ============================================================
    # Planejamento GLOBAL: mapa de ocupação (/grid_map) + A*  [Trab. 2]
    # ============================================================

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
        # Meta do planejador: a bandeira (se já estimada) senão o waypoint da zona azul.
        if self.flag_goal is not None:
            return self.flag_goal
        return self.search_goal

    def _replan(self):
        # Recalcula o caminho global robô→meta e publica /plan (RViz). Guarda
        # self.path (lista de (x,y) no mundo) p/ o seguidor de caminho (Etapa 3).
        if self.occ_grid is None or self.pose is None:
            return
        goal = self._plan_goal()
        if goal is None:
            return
        blocked = self._compute_blocked()
        start = self._nearest_free(blocked, self._world_to_cell(self.pose[0], self.pose[1]))
        if start is not None:
            blocked[start[1], start[0]] = False  # nunca bloqueia a célula do próprio robô
        gcell = self._nearest_free(blocked, self._world_to_cell(goal[0], goal[1]))
        if start is None or gcell is None:
            self.path = []
            self._publish_plan([])
            return
        cells = self._astar(blocked, start, gcell)
        self.path = self._simplify_path(cells) if cells else []
        self._publish_plan(self.path)

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
            # RETURNING_HOME: sem transição → estado terminal (fica parado).
        }

        #Estado inicial
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
        self.pose = None                    # (x, y, yaw) no frame odom_gt; None até 1ª msg
        self.flag_goal = None               # (x, y) estimado da bandeira no frame odom
        self.start_pose = None              # pose de spawn registrada (1ª msg de odom)
        self.search_goal = None             # (x, y) waypoint da busca dirigida (zona azul)
        # Planejamento global (Trab. 2): mapa de ocupação + caminho A*.
        self.occ_grid = None                # numpy (H,W) do /grid_map; None até 1ª msg
        self.map_info = None                # (res, ox, oy, W, H) do /grid_map
        self.path = []                      # caminho A* atual: lista de (x, y) no mundo
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

        # Planejamento global (Etapa 2): recalcula + publica /plan ~2x/s. Por ora só
        # PUBLICA o caminho (RViz); seguir o caminho é a Etapa 3.
        self._maybe_replan()

def main(args=None):
    rclpy.init(args=args)
    node = MissionFSM()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
