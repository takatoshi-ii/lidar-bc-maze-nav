"""記録・学習・推論で共通に使うLiDAR前処理と行動ラベルの定義"""
import numpy as np

# 行動クラス(teleop_node の w / a / d に対応)
ACTIONS = ['forward', 'left', 'right']
FORWARD, LEFT, RIGHT = 0, 1, 2

# 各行動の (linear.x, angular.z)。テレオペ(お手本)と自律走行で同じ値を使う
ACTION_TWIST = {
    FORWARD: (0.25, 0.0),
    LEFT: (0.08, 1.0),
    RIGHT: (0.08, -1.0),
}

DEFAULT_BINS = 72        # 360点 -> 72方向(5度刻み)
DEFAULT_CLIP = 3.0       # これより遠い距離は同一視する[m]


def preprocess_scan(ranges, range_max, n_bins=DEFAULT_BINS, clip=DEFAULT_CLIP):
    """LaserScan.ranges を n_bins 方向の最小距離に縮約し、0〜1に正規化する。

    ビン0が真正面(角度0)を中心にするよう回転させているので、
    mirror_bins() で左右反転できる。
    """
    r = np.asarray(ranges, dtype=np.float32)
    r = np.nan_to_num(r, nan=range_max, posinf=range_max, neginf=range_max)
    r = np.clip(r, 0.0, clip)
    width = max(len(r) // n_bins, 1)
    r = np.roll(r, width // 2)
    bins = np.array([chunk.min() for chunk in np.array_split(r, n_bins)], dtype=np.float32)
    return bins / clip


def mirror_bins(bins):
    """左右反転したスキャン(角度 θ -> -θ)を返す。データ拡張用。"""
    return np.roll(bins[..., ::-1], 1, axis=-1)


def mirror_label(label):
    """左右反転に合わせて left と right を入れ替える。"""
    label = np.asarray(label).copy()
    left = label == LEFT
    right = label == RIGHT
    label[left] = RIGHT
    label[right] = LEFT
    return label


def twist_to_label(linear_x, angular_z):
    """cmd_vel を行動クラスに変換する。停止・後退は学習に使わないので None。"""
    if angular_z > 1e-3:
        return LEFT
    if angular_z < -1e-3:
        return RIGHT
    if linear_x > 1e-3:
        return FORWARD
    return None
