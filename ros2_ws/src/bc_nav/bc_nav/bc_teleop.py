"""模倣学習のお手本用テレオペ(レースゲーム風)

  ロボットは自動で前進し、a / d を押している間だけ曲がる。
  w: 走行開始   スペース: 一時停止   q: 終了

ターミナルではキーを離したイベントが取れないので、キーリピートが
途切れたら「離した」とみなす。
"""
import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from bc_nav.scan_utils import ACTION_TWIST, FORWARD, LEFT, RIGHT

# 1回押したときに曲がる時間。1.0rad/sなら0.25秒で約14度
# (長押しはOSのキーリピートが始まるまで一瞬直進に戻る)
FIRST_PRESS_HOLD = 0.25
# リピート中はキー間隔(約0.03秒)より少し長く待てば十分
REPEAT_HOLD = 0.15

HELP = """
=== 模倣学習用テレオペ ===
  w       : 走行開始(自動で前進)
  a / d   : 押している間だけ 左 / 右 に曲がる
  スペース : 一時停止(記録されない)
  q       : 終了
いまは一時停止中です。w で走り始めます。
"""


class BCTeleop(Node):
    def __init__(self):
        super().__init__('bc_teleop')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def send(self, action):
        msg = Twist()
        if action is not None:
            msg.linear.x, msg.angular.z = ACTION_TWIST[action]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = BCTeleop()
    settings = termios.tcgetattr(sys.stdin)
    print(HELP)

    running = False
    steer = None          # LEFT / RIGHT / None(直進)
    last_key_time = 0.0
    prev_key_time = 0.0
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            now = time.monotonic()
            key = sys.stdin.read(1) if ready else None

            if key == 'q':
                break
            elif key == ' ':
                running, steer = False, None
                print('一時停止')
            elif key == 'w':
                if not running:
                    print('走行開始')
                running, steer = True, None
            elif key in ('a', 'd') and running:
                steer = LEFT if key == 'a' else RIGHT
                prev_key_time, last_key_time = last_key_time, now

            if steer is not None:
                repeating = last_key_time - prev_key_time < 0.1
                hold = REPEAT_HOLD if repeating else FIRST_PRESS_HOLD
                if now - last_key_time > hold:
                    steer = None

            if running:
                node.send(steer if steer is not None else FORWARD)
            else:
                node.send(None)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        if rclpy.ok():
            node.send(None)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
