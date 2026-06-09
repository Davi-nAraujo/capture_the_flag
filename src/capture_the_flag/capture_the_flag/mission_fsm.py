#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan, Imu, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped


from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from enum import Enum
class MissionFSM(Node):

    def _publish_cmd(self, lx, az):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'   # convention; controller doesn't enforce
        msg.twist.linear.x = lx
        msg.twist.angular.z = az
        self.cmd_vel_pub.publish(msg)

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
        self._publish_cmd(0.3, 0.0)


    def behavior_obstacle_avoidance(self):
        # Desvio em DUAS fases para escapar de mínimo local (obstáculo entre robô e
        # meta): (1) TURN gira no lugar para o lado livre; (2) ESCAPE anda PARA FRENTE
        # nessa direção livre, deslocando-se LATERALMENTE ao redor do obstáculo.
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
        # Gira no lugar para o lado onde a bandeira foi vista por último.
        # err<0 (esquerda) → CCW(+); err>0 (direita) → CW(-).
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
        half = self.image_width / 2.0
        err = (cx - half) / half               # ∈ [-1, +1]; + = bandeira à direita
        self.last_flag_err = err               # mantém o lado p/ REFINDING girar certo
        angular = -self.KP_BEARING * err       # gira para zerar o erro de bearing
        self._publish_cmd(0.0, angular)
        self.get_logger().info(
            f"ADJUSTING err={err:+.2f} area={self.latest_flag_area:.0f}",
            throttle_duration_sec=0.5)


    def behavior_collecting_flag(self):
        # Coleta simbólica (sem garra): fica PARADA junto à bandeira. A contagem
        # do tempo (e o log de sucesso) vivem na transição — o comportamento só
        # mantém o robô imóvel, fiel à divisão comportamento×transição do FSM.
        self._publish_cmd(0.0, 0.0)


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
    # Limiar de PERIGO (< FRONT_BLOCK_DIST): se algo está a menos que isto na frente,
    # nem a bandeira interrompe o desvio — a segurança vence. Acima dele (mas ainda
    # "bloqueado"), uma bandeira avistada durante o giro PODE travar a perseguição.
    # ~0.5 m do LIDAR = ~0.1 m da ponta do braço: imine, não vire para o alvo agora.
    DANGER_DIST = 0.5                     # m

    # OBSTACLE_AVOIDANCE — desvio COMPROMETIDO (anti-chatter), Lever 2:
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

    # COLLECTING_FLAG — coleta simbólica: fica parada N ticks junto à bandeira.
    # A 10 Hz, 50 ticks ≈ 5 s. Contamos TICKS (taxa fixa) pelo mesmo motivo do
    # CONFIRM_TICKS: mede tempo de verdade enquanto o tick for periódico.
    COLLECT_TICKS = 50

    # Servovisão do ajuste fino (ADJUSTING) + percepção da bandeira.
    KP_BEARING = 0.3       # ganho proporcional do erro horizontal (ADJUSTING)
    CENTER_TOL = 0.15      # |erro| abaixo disto = bandeira "centralizada"
    AREA_MIN = 10          # px²: blob menor que isto = ruído/longe demais → ignora
    AREA_NEAR = 3500       # px²: blob ≥ isto = perto o bastante p/ "chegou". Arrival
                           # VISUAL (a bandeira pode não estar no plano do LIDAR). TUNAR.
    FLAG_LABEL = 25        # blue_flag no labels_map (segmentação semântica)

    # REFINDING_FLAG — gira para reencontrar a bandeira perdida de vista.
    REFIND_ANGULAR = 0.5         # rad/s do giro de busca
    REFIND_TIMEOUT_TICKS = 120   # ~12 s (≈ uma volta) sem achar → desiste p/ EXPLORING

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

    def _range_at_bearing(self, bearing: float):
        # Lê o LIDAR no ângulo 'bearing' → (range, is_metric). Se o feixe não retorna
        # nada válido (bandeira além de range_max), devolve DEFAULT_FLAG_RANGE com
        # is_metric=False (só direção; refina ao entrar no alcance do LIDAR).
        scan = self.latest_scan
        if scan is None:
            return self.DEFAULT_FLAG_RANGE, False
        n = len(scan.ranges)
        idx = int(round(bearing / scan.angle_increment)) % n
        r = scan.ranges[idx]
        if math.isnan(r) or math.isinf(r) or r < scan.range_min or r > scan.range_max:
            return self.DEFAULT_FLAG_RANGE, False
        return r, True

    def _update_flag_goal(self, cx: float):
        # Projeta o centroide da bandeira no frame odom → ponto-meta (a MEMÓRIA).
        # Sem pose não há frame fixo onde ancorar; aborta silenciosamente.
        if self.pose is None or self.image_width is None:
            return
        x, y, yaw = self.pose
        half = self.image_width / 2.0
        err = (cx - half) / half                  # + = bandeira à direita
        bearing = -err * (self.CAM_HFOV / 2.0)    # à direita = bearing negativo (CW)
        rng, metric = self._range_at_bearing(bearing)
        self.flag_goal = (x + rng * math.cos(yaw + bearing),
                          y + rng * math.sin(yaw + bearing))
        self.flag_goal_metric = metric

    def _drive_to_point(self, gx: float, gy: float) -> float:
        # Controlador uniciclo: gira para o alvo e avança proporcional à distância,
        # freando quando desalinhado. Retorna a distância restante (inf se sem pose).
        if self.pose is None:
            self._publish_cmd(0.0, 0.0)
            return float('inf')
        x, y, yaw = self.pose
        dist = math.hypot(gx - x, gy - y)
        berr = self._norm_angle(math.atan2(gy - y, gx - x) - yaw)
        ang = max(-self.GOAL_W_MAX, min(self.GOAL_W_MAX, self.GOAL_KP_ANG * berr))
        lin = self.GOAL_KP_LIN * dist
        lin *= max(0.0, 1.0 - abs(berr) / self.GOAL_ALIGN_TOL)  # só avança alinhado
        lin = max(0.0, min(self.GOAL_V_MAX, lin))
        self._publish_cmd(lin, ang)
        return dist

    def _flag_centered(self) -> bool:
        centroid = self.latest_flag_centroid
        if centroid is None or self.image_width is None:
            return False
        cx, _ = centroid
        half = self.image_width / 2.0
        return abs((cx - half) / half) < self.CENTER_TOL

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
        # 3) Nada de interessante: continua explorando.
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
        # 4) Segue navegando para a meta.
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
            self.collect_ticks = 0   # zera o relógio de coleta ao ENTRAR
            return self.States.COLLECTING_FLAG
        # 3) Ainda torta: continua girando para centralizar.
        return None

    def transition_collecting_flag(self):
        # Rede de segurança (v5): perdeu a bandeira durante a coleta? Reprocura.
        # Raro — parada, centralizada e perto (área enorme) — mas fiel ao diagrama.
        if self.latest_flag_centroid is None:
            self.refind_ticks = 0
            return self.States.REFINDING_FLAG
        # Conta o tempo imóvel junto à bandeira. ~5 s a 10 Hz = 50 ticks.
        self.collect_ticks += 1
        if self.collect_ticks >= self.COLLECT_TICKS:
            self.get_logger().info(
                "BANDEIRA COLETADA! Missao cumprida. -> RETURNING_HOME")
            return self.States.RETURNING_HOME
        return None  # ainda coletando: fica parada

    # RETURNING_HOME não tem transição: é estado FINAL (missão concluída, robô parado).

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
        self.collect_ticks = 0              # ticks parada coletando a bandeira
        self.avoid_ticks = 0                # ticks no desvio atual (compromisso mínimo)
        self.avoid_turn_dir = 1.0           # sentido travado do giro de desvio (+esq/-dir)
        self.avoid_phase = 'TURN'           # fase do desvio: 'TURN' (gira) ou 'ESCAPE' (anda)
        self.escape_ticks = 0               # ticks andando na fase ESCAPE
        self.pose = None                    # (x, y, yaw) no frame odom_gt; None até 1ª msg
        self.flag_goal = None               # (x, y) estimado da bandeira no frame odom
        self.flag_goal_metric = False       # True se a meta veio de range REAL do LIDAR

        # Publisher para comando de velocidade
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/diff_drive_base_controller/cmd_vel', 10)

        # Subscribers
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(Odometry, '/odom_gt', self.odom_callback, 10)
        self.create_subscription(Image, '/robot_cam/labels_map', self.camera_callback, 10)

        # Utilizado para converter imagens ROS -> OpenCV
        self.bridge = CvBridge()

        # Timer para enviar comandos continuamente
        self.timer = self.create_timer(0.1, self.tick)

        # Estado interno
        # self.obstaculo_a_frente = False

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg
        # # Verifica uma faixa estreita ao redor de 0° (frente)
        # num_ranges = len(msg.ranges)
        # if num_ranges == 0:
        #     return

        # # Índices de -30° a +30° (equivalente a 330 até 30)
        # indices_frente = list(range(330, 360)) + list(range(0, 31))

        # # Filtra distancias
        # distancias = [msg.ranges[i] for i in indices_frente]

        # if distancias and min(distancias) < 0.5:
        #     self.obstaculo_a_frente = True
        #     self.get_logger().info('Obstáculo detectado a {:.2f}m à frente'.format(min(distancias)))
        # else:
        #     self.obstaculo_a_frente = False

    def imu_callback(self, msg: Imu):
        # # Extraindo o quaternion da mensagem
        # orientation_q = msg.orientation
        # quat = [
        #     orientation_q.x,
        #     orientation_q.y,
        #     orientation_q.z,
        #     orientation_q.w
        # ]

        # # Conversão para Euler usando SciPy
        # r = R.from_quat(quat)
        # roll, pitch, yaw = r.as_euler('xyz', degrees=True)

        # # Exibindo resultados
        # self.get_logger().info('IMU Data Received:')
        # self.get_logger().info(
        #     f'Orientation (Euler): Roll={roll:.2f}°, '
        #     f'Pitch={pitch:.2f}°, Yaw={yaw:.2f}°'
        # )
        # self.get_logger().info(
        #     f'Angular velocity: [{msg.angular_velocity.x:.2f}, '
        #     f'{msg.angular_velocity.y:.2f}, {msg.angular_velocity.z:.2f}] rad/s'
        # )
        # self.get_logger().info(
        #     f'Linear acceleration: [{msg.linear_acceleration.x:.2f}, '
        #     f'{msg.linear_acceleration.y:.2f}, {msg.linear_acceleration.z:.2f}] m/s²'
        # )
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

def main(args=None):
    rclpy.init(args=args)
    node = MissionFSM()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
