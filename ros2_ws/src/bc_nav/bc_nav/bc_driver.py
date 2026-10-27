"""学習済みモデルで /scan から行動を推論し /cmd_vel を出す自律走行ノード

AIの判断に加えて、2つの補助を入れている。
  - 平滑化: 行動確率を指数移動平均し、左右の判断がフレームごとに入れ替わるのを防ぐ
  - リカバリー: 一定時間ほとんど進めていなければ、バックして開けた方向へ向き直す
    (お手本に「バック」が無いため、AIだけでは引っかかりから抜け出せない)
"""
import math
import os
import time

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from bc_nav.scan_utils import preprocess_scan, ACTIONS, ACTION_TWIST
from bc_nav.train_bc import build_model


class BCDriver(Node):
    def __init__(self):
        super().__init__('bc_driver')
        self.declare_parameter('model_path', os.path.expanduser('~/bc_nav_data/model.pt'))
        self.declare_parameter('smoothing', 0.5)      # 0で平滑化なし。大きいほど判断が安定する
        self.declare_parameter('stuck_time', 3.0)     # この秒数で
        self.declare_parameter('stuck_dist', 0.05)    # これ以下しか進まなければ引っかかりと判断[m]
        self.declare_parameter('backup_time', 1.0)
        self.declare_parameter('turn_time', 1.2)
        self.declare_parameter('duration', 0.0)       # 評価用: この秒数で自動停止(0なら無制限)
        self.declare_parameter('results_file', os.path.expanduser('~/bc_nav_data/eval_results.csv'))
        self.declare_parameter('start_x', 0.0)        # 結果の記録用(スタート位置)
        self.declare_parameter('start_y', 0.0)

        ckpt = torch.load(self.get_parameter('model_path').value, map_location='cpu')
        self.n_bins = ckpt['n_bins']
        self.clip = ckpt['clip']
        self.model = build_model(self.n_bins)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

        self.smoothing = self.get_parameter('smoothing').value
        self.stuck_time = self.get_parameter('stuck_time').value
        self.stuck_dist = self.get_parameter('stuck_dist').value
        self.backup_time = self.get_parameter('backup_time').value
        self.turn_time = self.get_parameter('turn_time').value
        self.duration = self.get_parameter('duration').value

        # 評価用の記録
        self.start_time = None
        self.last_time = None
        self.finished = False
        self.distance = 0.0
        self.min_range = float('inf')
        self.self_mask = None       # LiDARが自分の車体を見ている方向(最接近距離の計算から除く)

        self.probs = None
        self.last_action = None
        self.pose = None
        self.anchor = None          # (時刻, x, y) 進んだかどうかの基準点
        self.recovery_until = 0.0
        self.recovery_turn = 0.0
        self.recovery_count = 0

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        # 動画に重ねる表示用(video_recorder が購読する)
        self.status_pub = self.create_publisher(String, '/bc_status', 10)
        self.get_logger().info('自律走行を開始します')

    def publish_status(self, text):
        self.status_pub.publish(String(data=text))

    def odom_callback(self, msg):
        pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if self.pose is not None and self.start_time is not None and not self.finished:
            self.distance += math.hypot(pose[0] - self.pose[0], pose[1] - self.pose[1])
        self.pose = pose

    def elapsed(self):
        return 0.0 if self.start_time is None else self.last_time - self.start_time

    def save_results(self):
        """評価結果をCSVに1行追記する。走っていなければ何もしない。"""
        elapsed = self.elapsed()
        if elapsed <= 0:
            return None
        path = self.get_parameter('results_file').value
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new_file = not os.path.exists(path)
        with open(path, 'a', encoding='utf-8') as f:
            if new_file:
                f.write('datetime,model,start_x,start_y,time_s,distance_m,'
                        'speed_mps,recoveries,min_range_m\n')
            f.write(','.join([
                time.strftime('%Y-%m-%d %H:%M:%S'),
                os.path.basename(self.get_parameter('model_path').value),
                f"{self.get_parameter('start_x').value:.2f}",
                f"{self.get_parameter('start_y').value:.2f}",
                f'{elapsed:.1f}',
                f'{self.distance:.2f}',
                f'{self.distance / elapsed:.3f}',
                str(self.recovery_count),
                f'{self.min_range:.3f}',
            ]) + '\n')
        return path

    def summary(self):
        elapsed = self.elapsed()
        speed = self.distance / elapsed if elapsed > 0 else 0.0
        return ('\n===== 評価結果 =====\n'
                f'走行時間       : {elapsed:.1f} 秒\n'
                f'走行距離       : {self.distance:.2f} m\n'
                f'平均速度       : {speed:.3f} m/s\n'
                f'リカバリー回数 : {self.recovery_count} 回\n'
                f'壁への最接近   : {self.min_range:.3f} m\n'
                '====================')

    def is_stuck(self, now):
        if self.pose is None:
            return False
        if self.anchor is None:
            self.anchor = (now, *self.pose)
            return False
        t0, x0, y0 = self.anchor
        if math.hypot(self.pose[0] - x0, self.pose[1] - y0) > self.stuck_dist:
            self.anchor = (now, *self.pose)
            return False
        return now - t0 > self.stuck_time

    def scan_callback(self, msg):
        if self.finished:
            return
        # 時間はすべてシミュレーション時刻(スキャンのタイムスタンプ)で測る。
        # 録画などでシミュレーションが実時間より遅くなっても、走行時間や引っかかり判定がずれない
        now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.last_time = now
        if self.start_time is None:
            self.start_time = now
        if self.duration > 0 and now - self.start_time >= self.duration:
            self.finished = True
            self.stop()
            self.publish_status('AI: finished')
            return

        ranges = np.asarray(msg.ranges, dtype=np.float32)
        if self.self_mask is None:
            # スタート位置は壁から0.4m以上離れているので、最初のスキャンで0.3mより近いものは自分の車体
            self.self_mask = ranges < 0.3
            self.get_logger().info(f'自分の車体が見えている方向: {int(self.self_mask.sum())}本 (最接近距離の計算から除外)')
        # range_min ちょうどの値は「測定範囲より近い」という意味なので距離としては使わない
        valid = ranges[np.isfinite(ranges) & (ranges > msg.range_min + 0.005) & ~self.self_mask]
        if valid.size:
            self.min_range = min(self.min_range, float(valid.min()))

        x = preprocess_scan(msg.ranges, msg.range_max, self.n_bins, self.clip)
        cmd = Twist()

        if now < self.recovery_until:
            # 前半はバック、後半は開けている方向へその場旋回
            if self.recovery_until - now > self.turn_time:
                cmd.linear.x = -0.15
                self.publish_status(f'RECOVERY #{self.recovery_count}: back')
            else:
                cmd.angular.z = self.recovery_turn
                self.publish_status(f'RECOVERY #{self.recovery_count}: turn')
            self.pub.publish(cmd)
            return

        if self.is_stuck(now):
            half = self.n_bins // 2
            left_open = x[1:half].mean() > x[half + 1:].mean()
            self.recovery_turn = 1.0 if left_open else -1.0
            self.recovery_until = now + self.backup_time + self.turn_time
            self.recovery_count += 1
            self.anchor = None
            self.probs = None
            self.get_logger().warn(f'引っかかりを検出 -> リカバリー ({self.recovery_count}回目)')
            return

        with torch.no_grad():
            p = torch.softmax(self.model(torch.from_numpy(x).unsqueeze(0)), dim=1)[0].numpy()
        self.probs = p if self.probs is None else self.smoothing * self.probs + (1 - self.smoothing) * p
        action = int(np.argmax(self.probs))

        # お手本(bc_teleop)と同じ速度で動かす
        cmd.linear.x, cmd.angular.z = ACTION_TWIST[action]
        self.pub.publish(cmd)
        self.publish_status(f'AI: {ACTIONS[action]}  (p={self.probs[action]:.2f})')

        if action != self.last_action:
            self.get_logger().info(f'行動: {ACTIONS[action]}')
            self.last_action = action

    def stop(self):
        self.pub.publish(Twist())


def main():
    rclpy.init()
    node = BCDriver()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        print(node.summary(), flush=True)
        path = node.save_results()
        if path:
            print(f'[bc_driver] 結果を追記しました: {path}', flush=True)
        if rclpy.ok():
            node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
