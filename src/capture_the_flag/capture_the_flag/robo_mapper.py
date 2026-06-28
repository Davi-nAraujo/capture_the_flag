#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan, Imu, Image
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Twist, Pose

from std_msgs.msg import Header

from scipy.spatial.transform import Rotation as R

from cv_bridge import CvBridge
import cv2
import numpy as np
import math

# Necessario para publicar o frame map:
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped


class RoboMapper(Node):
    # ------------------------------------------------------------------
    # Mapa de ocupação por LOG-ODDS, cobrindo TODA a arena (18x8 m).
    # Como /odom_gt (pose do modelo) é GROUND TRUTH sem deriva, isto é
    # "mapeamento com pose conhecida" — não é SLAM: basta carimbar cada
    # retorno do LIDAR na grade na pose exata do robô.
    # ------------------------------------------------------------------
    RESOLUTION = 0.1            # m por célula
    ORIGIN_X = -9.5            # mundo: canto inferior-esquerdo do grid (x)
    ORIGIN_Y = -4.5            # mundo: canto inferior-esquerdo do grid (y)
    WIDTH = 190                # células em x  -> cobre x ∈ [-9.5, +9.5]
    HEIGHT = 90                # células em y  -> cobre y ∈ [-4.5, +4.5]

    # Log-odds: cada observação ajusta a "crença" de ocupação da célula.
    L_FREE = -0.4              # célula ATRAVESSADA pelo feixe -> mais livre
    L_OCC = 0.85              # célula no FIM do feixe -> mais ocupada
    L_MIN, L_MAX = -4.0, 4.0   # saturação (permite revisar, evita certeza absoluta)
    OCC_THRESH = 0.5          # log-odds > isto  -> ocupado (100)
    FREE_THRESH = -0.5         # log-odds < isto  -> livre (0); entre os dois -> -1 (desconhecido)

    INTEGRATE_PERIOD = 0.2     # s: integra no máximo ~5x/s (suficiente p/ robô lento)

    def __init__(self):
        super().__init__('robo_mapper')

        # Subscribers
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Pose, '/model/prm_robot/pose', self.odom_callback, 10)
        self.create_subscription(Image, '/robot_cam/colored_map', self.camera_callback, 10)

        # Utilizado para converter imagens ROS -> OpenCV
        self.bridge = CvBridge()

        # Timer só para PUBLICAR o mapa (a integração acontece no scan_callback).
        self.timer = self.create_timer(0.5, self.atualiza_mapa)

        # Estado atual do robô (pose ground-truth):
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.have_pose = False
        self.last_integrate = 0.0

        # Mapa em LOG-ODDS (float). 0 = desconhecido; sobe p/ ocupado, desce p/ livre.
        self.logodds = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.float32)

        # Publisher do mapa
        self.map_pub = self.create_publisher(OccupancyGrid, '/grid_map', 10)

        # TF estático map -> odom_gt (coincidem: odom_gt já é o mundo ground-truth).
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        static_tf = TransformStamped()
        static_tf.header.stamp = self.get_clock().now().to_msg()
        static_tf.header.frame_id = "map"
        static_tf.child_frame_id = "odom_gt"
        static_tf.transform.translation.x = 0.0
        static_tf.transform.translation.y = 0.0
        static_tf.transform.translation.z = 0.0
        static_tf.transform.rotation.w = 1.0  # identidade (Quaternions!!)
        self.tf_static_broadcaster.sendTransform(static_tf)

    # ------------------------------------------------------------------
    # Conversões mundo <-> grid
    # ------------------------------------------------------------------
    def world_to_grid(self, x, y):
        gx = int((x - self.ORIGIN_X) / self.RESOLUTION)
        gy = int((y - self.ORIGIN_Y) / self.RESOLUTION)
        return gx, gy

    def in_bounds(self, gx, gy):
        return 0 <= gx < self.WIDTH and 0 <= gy < self.HEIGHT

    # ------------------------------------------------------------------
    # Pose ground-truth do robô (mesma fonte do /odom_gt)
    # ------------------------------------------------------------------
    def odom_callback(self, msg: Pose):
        self.x = msg.position.x
        self.y = msg.position.y
        q = msg.orientation
        self.heading = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz', degrees=False)[2]
        self.have_pose = True

    # ------------------------------------------------------------------
    # Integração da varredura LIDAR no mapa (uma vez a cada INTEGRATE_PERIOD).
    # Para cada feixe: traça o raio do robô até o ponto medido marcando as
    # células do caminho como LIVRES e a célula final como OCUPADA (se houve
    # retorno real; alcance máximo = "nada ali", então só limpa, sem obstáculo).
    # ------------------------------------------------------------------
    def scan_callback(self, msg: LaserScan):
        if not self.have_pose:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_integrate < self.INTEGRATE_PERIOD:
            return
        self.last_integrate = now

        x0, y0, th = self.x, self.y, self.heading
        res = self.RESOLUTION
        inv_res = 1.0 / res
        ox, oy = self.ORIGIN_X, self.ORIGIN_Y
        W, H = self.WIDTH, self.HEIGHT
        lo = self.logodds
        max_r = msg.range_max
        step = res                 # passo de amostragem ao longo do raio

        n = len(msg.ranges)
        for i in range(n):
            r = msg.ranges[i]
            if math.isnan(r):
                continue
            hit = True
            if math.isinf(r) or r >= max_r:
                r = max_r
                hit = False        # sem retorno: raio livre até o alcance, sem obstáculo
            if r < msg.range_min:
                continue
            ang = th + msg.angle_min + i * msg.angle_increment
            ca = math.cos(ang)
            sa = math.sin(ang)

            # Células LIVRES ao longo do raio (até pouco antes do fim).
            d = step
            r_free = r - res
            while d < r_free:
                gx = int((x0 + d * ca - ox) * inv_res)
                gy = int((y0 + d * sa - oy) * inv_res)
                if 0 <= gx < W and 0 <= gy < H:
                    lo[gy, gx] += self.L_FREE
                d += step

            # Célula final OCUPADA (apenas em retorno real).
            if hit:
                gx = int((x0 + r * ca - ox) * inv_res)
                gy = int((y0 + r * sa - oy) * inv_res)
                if 0 <= gx < W and 0 <= gy < H:
                    lo[gy, gx] += self.L_OCC

        np.clip(lo, self.L_MIN, self.L_MAX, out=lo)

    def camera_callback(self, msg: Image):
        # Não usado no mapeamento de ocupação (só LIDAR). Mantido p/ o sensor.
        pass

    # ------------------------------------------------------------------
    # Publicação periódica do mapa
    # ------------------------------------------------------------------
    def atualiza_mapa(self):
        self.publish_occupancy_grid()

    def publish_occupancy_grid(self):
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = "map"

        # Metadados do mapa
        grid_msg.info.resolution = self.RESOLUTION
        grid_msg.info.width = self.WIDTH
        grid_msg.info.height = self.HEIGHT

        # Origem do mapa (canto inferior esquerdo do grid no mundo)
        origin = Pose()
        origin.position.x = self.ORIGIN_X
        origin.position.y = self.ORIGIN_Y
        origin.position.z = 0.0
        origin.orientation.w = 1.0
        grid_msg.info.origin = origin

        # log-odds -> ocupação padrão: 0 livre, 100 ocupado, -1 desconhecido
        occ = np.full(self.logodds.shape, -1, dtype=np.int8)
        occ[self.logodds > self.OCC_THRESH] = 100
        occ[self.logodds < self.FREE_THRESH] = 0
        grid_msg.data = occ.flatten().tolist()

        # Publicar
        self.map_pub.publish(grid_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RoboMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
