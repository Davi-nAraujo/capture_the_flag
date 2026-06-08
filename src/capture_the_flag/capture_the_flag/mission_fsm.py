#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan, Imu, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
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
        self._publish_cmd(0.2, 0.0)


    def behavior_obstacle_avoidance(self):
        self._publish_cmd(0.0, 0.5)

        
    def behavior_flag_found_confirmation(self):
        self._publish_cmd(0.0, 0.0)


    def behavior_going_to_flag(self):
        centroid = self.latest_flag_centroid
        if centroid is None or self.image_width is None:
            self._publish_cmd(0.0, 0.0)  # sem alvo; a transição já trata disso
            return
        cx, _ = centroid
        half = self.image_width / 2.0
        err = (cx - half) / half               # ∈ [-1, +1]; + = bandeira à direita
        angular = -self.KP_BEARING * err        # gira para reduzir o erro de bearing
        forward = self.FWD_TO_FLAG * max(0.0, 1.0 - abs(err))  # freia se desalinhado
        self._publish_cmd(forward, angular)
        # Log para tunar AREA_NEAR (veja a área crescer ao se aproximar) e Kp.
        self.get_logger().info(
            f"GOING_TO_FLAG err={err:+.2f} area={self.latest_flag_area:.0f}",
            throttle_duration_sec=0.5)


    def behavior_refinding_flag(self):
        self._publish_cmd(0.0, 0.0)


    def behavior_adjusting_position(self):
        self._publish_cmd(0.0, 0.0)


    def behavior_collecting_flag(self):
        self._publish_cmd(0.0, 0.0)


    def behavior_returning_home(self):
        self._publish_cmd(0.0, 0.0)


    # --- Percepção: helpers de LIDAR ---

    # LIDAR diagnosticado via `ros2 topic echo /scan --once`:
    # angle_min=0, angle_max≈2π, ~1°/raio, range=[0.12, 3.5]. Índice 0 = FRENTE.
    FRONT_ARC_HALF = math.radians(25.0)   # meia-largura do arco frontal (rad)
    FRONT_BLOCK_DIST = 0.5                 # m: distância que conta como "bloqueado"

    # Confirmação da bandeira: nº de ticks consecutivos com a bandeira visível
    # antes de aceitar. A 10 Hz, 20 ticks ≈ 2 s. Contamos TICKS (taxa fixa), não
    # frames da câmera — por isso isto mede tempo de verdade. Se o tick virasse
    # variável, o robusto seria medir com get_clock().now().
    CONFIRM_TICKS = 20

    # GOING_TO_FLAG — servovisão proporcional sobre o erro de bearing.
    KP_BEARING = 0.5       # ganho proporcional do erro horizontal normalizado
    FWD_TO_FLAG = 0.2      # m/s à frente quando bem alinhado
    CENTER_TOL = 0.15      # |erro| abaixo disto = bandeira "centralizada"
    AREA_NEAR = 4000       # px²: blob maior que isto = perto (TUNAR vendo rodar)
    FLAG_LABEL = 25        # blue_flag no labels_map (segmentação semântica)

    def _front_blocked(self) -> bool:
        scan = self.latest_scan
        if scan is None:
            return True  # sem dados ainda → trate como bloqueado (seguro)

        k = int(round(self.FRONT_ARC_HALF / scan.angle_increment))
        # Frente = índice 0; janela embrulha: 0..k (à esquerda) e n-k..n-1 (à direita).
        window = list(scan.ranges[:k + 1]) + list(scan.ranges[-k:])

        # inf = nada detectado no alcance = LIVRE (mantém). Descarta só inválidos:
        # NaN e leituras abaixo do range_min (os 0.0 que aparecem no echo).
        valid = [r for r in window if not math.isnan(r) and r >= scan.range_min]
        if not valid:
            return True  # cego de verdade (tudo NaN) → bloqueado por segurança
        return min(valid) < self.FRONT_BLOCK_DIST

    def _flag_centered(self) -> bool:
        centroid = self.latest_flag_centroid
        if centroid is None or self.image_width is None:
            return False
        cx, _ = centroid
        half = self.image_width / 2.0
        return abs((cx - half) / half) < self.CENTER_TOL

    def _arrived_at_flag(self) -> bool:
        # Perto = blob grande (proxy de distância por visão). Sem profundidade,
        # área é o único sinal de proximidade confiável: "bloqueado + centralizado"
        # confunde a bandeira com um obstáculo que a tem centralizada atrás.
        # Tunar AREA_NEAR observando o log de área durante a aproximação.
        return (self.latest_flag_area is not None
                and self.latest_flag_area >= self.AREA_NEAR)

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
            self.previous_state = self.current_state  # registro p/ retornar depois
            return self.States.OBSTACLE_AVOIDANCE
        # 2) Bandeira visível? Vai confirmar.
        if self.latest_flag_centroid is not None:
            self.flag_seen_consecutive = 0  # zera o contador ao ENTRAR na confirmação
            return self.States.FLAG_FOUND_CONFIRMATION
        # 3) Nada de interessante: continua explorando.
        return None

    def transition_obstacle_avoidance(self):
        # Frente liberou? Volta para quem chamou (EXPLORING, GOING_TO_FLAG, ...).
        if not self._front_blocked():
            return self.previous_state
        return None  # ainda bloqueado: continua girando

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
        # 1) Perdeu a bandeira de vista? Procura de novo.
        if self.latest_flag_centroid is None:
            return self.States.REFINDING_FLAG
        # 2) Chegou? Vai posicionar para coletar — ANTES do desvio agarrar a
        #    bandeira (a chegada vence o empate objetivo/obstáculo no alvo).
        if self._arrived_at_flag():
            return self.States.ADJUSTING_POSITION_TO_COLLECT_FLAG
        # 3) Obstáculo à frente que NÃO é a bandeira (descentralizado)? Desvia.
        if self._front_blocked():
            self.previous_state = self.current_state
            return self.States.OBSTACLE_AVOIDANCE
        # 4) Segue mirando.
        return None

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

        # Publisher para comando de velocidade
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/diff_drive_base_controller/cmd_vel', 10)

        # Subscribers
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
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
        # Mensagens de Odometria das rodas!
        pass

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

        # DEBUG (remover após confirmar): quais labels estão no quadro agora.
        self.get_logger().info(
            f"labels em vista: {np.unique(label_img).tolist()}",
            throttle_duration_sec=1.0)

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
        M = cv2.moments(largest)
        if M['m00'] == 0:
            self.latest_flag_centroid = None
            self.latest_flag_area = None
            return

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        self.latest_flag_centroid = (cx, cy)
        self.latest_flag_area = cv2.contourArea(largest)

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
