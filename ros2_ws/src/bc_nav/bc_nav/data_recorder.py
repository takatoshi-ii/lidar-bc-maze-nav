"""テレオペ中の /scan と /cmd_vel をペアで記録するノード

teleop_node はキーを押した瞬間にだけ /cmd_vel を出し、ロボットは最後の指令を
保ち続けるので、「直近に受け取った cmd_vel」をそのスキャン時点の行動とみなす。
"""
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

from bc_nav.scan_utils import preprocess_scan, twist_to_label, ACTIONS


class DataRecorder(Node):
    def __init__(self):
        super().__init__('data_recorder')
        self.declare_parameter('data_dir', os.path.expanduser('~/bc_nav_data'))
        self.declare_parameter('every_n_scans', 3)     # 約30Hz -> 約10Hzに間引く
        self.declare_parameter('autosave_every', 200)  # 異常終了に備えて定期保存

        data_dir = self.get_parameter('data_dir').value
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, time.strftime('session_%Y%m%d_%H%M%S.npz'))
        self.every_n = self.get_parameter('every_n_scans').value
        self.autosave_every = self.get_parameter('autosave_every').value

        self.last_label = None
        self.scan_count = 0
        self.scans = []
        self.labels = []

        self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.get_logger().info(f'記録開始: {self.path}')

    def cmd_callback(self, msg):
        self.last_label = twist_to_label(msg.linear.x, msg.angular.z)

    def scan_callback(self, msg):
        self.scan_count += 1
        if self.scan_count % self.every_n != 0 or self.last_label is None:
            return
        self.scans.append(preprocess_scan(msg.ranges, msg.range_max))
        self.labels.append(self.last_label)

        if len(self.labels) % self.autosave_every == 0:
            self.save()
            counts = np.bincount(self.labels, minlength=len(ACTIONS))
            summary = ', '.join(f'{a}={c}' for a, c in zip(ACTIONS, counts))
            self.get_logger().info(f'{len(self.labels)} サンプル ({summary})')

    def save(self):
        if not self.labels:
            return
        np.savez(self.path, scans=np.stack(self.scans), labels=np.array(self.labels))


def main():
    rclpy.init()
    node = DataRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        print(f'[data_recorder] {len(node.labels)} サンプルを保存: {node.path}')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
