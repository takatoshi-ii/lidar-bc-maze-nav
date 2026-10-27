"""真上カメラの映像に情報を重ねてMP4に保存するノード

重ねる情報: タイトル、経過時間、現在の行動(HUMAN/AI)、リカバリー中の表示、走行軌跡
保存先: ~/bc_nav_data/videos/<prefix>_<日時>.mp4
"""
import math
import os
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from bc_nav.scan_utils import ACTIONS, twist_to_label

STATUS_TIMEOUT = 0.5   # これより古い /bc_status は使わない[秒]


def load_wall_tops(stl_path):
    """バイナリSTLから壁の上面(一番高い水平面)の三角形を (x, y) で返す"""
    with open(stl_path, 'rb') as f:
        data = f.read()
    n = int(np.frombuffer(data[80:84], dtype='<u4')[0])
    faces = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype(
        [('normal', '<3f4'), ('verts', '<9f4'), ('attr', '<u2')]))
    tris = faces['verts'].reshape(-1, 3, 3)
    z = tris[:, :, 2].mean(axis=1)
    top = (faces['normal'][:, 2] > 0.99) & (np.abs(z - z.max()) < 0.005)
    return tris[top][:, :, :2]


class VideoRecorder(Node):
    def __init__(self):
        super().__init__('video_recorder')
        self.declare_parameter('output_dir', os.path.expanduser('~/bc_nav_data/videos'))
        self.declare_parameter('prefix', 'run')
        self.declare_parameter('title', '')
        self.declare_parameter('actor', 'HUMAN')    # 誰が操作しているか(HUMAN / AI)
        self.declare_parameter('fps', 15.0)
        # 真上カメラの設置条件(bc_world.sdf と合わせる)。軌跡を画像に描くのに使う
        self.declare_parameter('camera_x', 3.46)
        self.declare_parameter('camera_y', 3.46)
        self.declare_parameter('camera_height', 6.8)
        self.declare_parameter('camera_hfov', 1.0)
        # 真上からだと壁の上面が床と同じ色で見えないので、形状データから壁を描き足す
        self.declare_parameter('field_stl', os.path.join(
            get_package_share_directory('nav_dev'),
            'models', 'field', 'meshes', 'HaruRobo2024field.stl'))

        output_dir = self.get_parameter('output_dir').value
        os.makedirs(output_dir, exist_ok=True)
        prefix = self.get_parameter('prefix').value
        self.path = os.path.join(output_dir, time.strftime(f'{prefix}_%Y%m%d_%H%M%S.mp4'))
        self.title = self.get_parameter('title').value
        self.actor = self.get_parameter('actor').value
        self.fps = self.get_parameter('fps').value
        self.cam_x = self.get_parameter('camera_x').value
        self.cam_y = self.get_parameter('camera_y').value
        self.cam_h = self.get_parameter('camera_height').value
        self.hfov = self.get_parameter('camera_hfov').value
        self.wall_tris = load_wall_tops(self.get_parameter('field_stl').value)
        self.wall_mask = None

        self.writer = None
        self.frames = 0
        self.start_stamp = None
        self.action_text = ''
        self.status_text = ''
        self.status_time = 0.0
        self.trail = []

        self.create_subscription(Image, '/overhead_camera', self.image_callback, 10)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        self.create_subscription(String, '/bc_status', self.status_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.get_logger().info(f'録画先: {self.path}')

    def cmd_callback(self, msg):
        label = twist_to_label(msg.linear.x, msg.angular.z)
        self.action_text = 'STOP' if label is None else ACTIONS[label]

    def status_callback(self, msg):
        self.status_text = msg.data
        self.status_time = time.monotonic()

    def odom_callback(self, msg):
        # このロボットの /odom はワールド座標(スタート地点が原点ではない)
        p = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if not self.trail or math.hypot(p[0] - self.trail[-1][0], p[1] - self.trail[-1][1]) > 0.02:
            self.trail.append(p)

    def to_pixel(self, x, y, width, height):
        # 画像の上がワールドの+x、左が+y (カメラは真下向き、高さ camera_height)
        f = (width / 2) / math.tan(self.hfov / 2)
        u = width / 2 - (y - self.cam_y) * f / self.cam_h
        v = height / 2 - (x - self.cam_x) * f / self.cam_h
        return int(round(u)), int(round(v))

    def image_callback(self, msg):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        frame = frame[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, (msg.width, msg.height))
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.start_stamp is None:
            self.start_stamp = stamp

        self.draw_overlay(frame, stamp - self.start_stamp)
        self.writer.write(frame)
        self.frames += 1

    def draw_overlay(self, frame, elapsed):
        h, w = frame.shape[:2]
        if self.wall_mask is None:
            self.wall_mask = np.zeros((h, w), dtype=np.uint8)
            for tri in self.wall_tris:
                pts = np.array([self.to_pixel(x, y, w, h) for x, y in tri], dtype=np.int32)
                cv2.fillPoly(self.wall_mask, [pts], 255)
        frame[self.wall_mask > 0] = (70, 70, 70)

        if len(self.trail) > 1:
            pts = np.array([self.to_pixel(x, y, w, h) for x, y in self.trail], dtype=np.int32)
            cv2.polylines(frame, [pts], False, (0, 140, 255), 1, cv2.LINE_AA)

        status_fresh = time.monotonic() - self.status_time < STATUS_TIMEOUT
        if status_fresh and self.status_text.startswith('RECOVERY'):
            text, color = self.status_text, (0, 0, 255)
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 8)
        elif status_fresh:
            text, color = self.status_text, (0, 120, 0)
        else:
            text, color = f'{self.actor}: {self.action_text}', (160, 60, 0)

        lines = [(self.title, (40, 40, 40)), (f'time {elapsed:6.1f} s', (40, 40, 40)), (text, color)]
        # ロボットが画像の上半分(区画B)にいるときは、テロップと重ならないよう下に出す
        top = 34
        if self.trail and self.to_pixel(*self.trail[-1], w, h)[1] < h / 2:
            top = h - 20 - (len(lines) - 1) * 34
        for i, (line, c) in enumerate(lines):
            if not line:
                continue
            y = top + i * 34
            cv2.putText(frame, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 5, cv2.LINE_AA)
            cv2.putText(frame, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2, cv2.LINE_AA)

    def close(self):
        if self.writer is not None:
            self.writer.release()


def main():
    rclpy.init()
    node = VideoRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        seconds = node.frames / node.fps if node.fps else 0
        print(f'[video_recorder] {node.frames} フレーム({seconds:.1f}秒)を保存: {node.path}', flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
