"""正方形などの動画の左右に余白を付けて横長(16:9)にする

YouTube は正方形・縦長で3分以内の動画を Shorts 扱いにし、左右を切って表示する。
横長にすれば通常の動画として扱われ、チャプターも使える。

使い方 (Windows の Python で実行):
  python tools/pad_to_16x9.py 入力.mp4 出力.mp4
"""
import argparse

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--bg', default='#202020', help='余白の色')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f'動画を開けません: {args.input}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = max(width, (height * 16 // 9 + 1) // 2 * 2)   # 偶数にそろえる
    x0 = (out_w - width) // 2

    rgb = tuple(int(args.bg.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
    canvas = np.zeros((height, out_w, 3), dtype=np.uint8)
    canvas[:] = rgb[::-1]   # BGR
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, height))

    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        canvas[:, x0:x0 + width] = frame
        writer.write(canvas)
        count += 1
    cap.release()
    writer.release()
    print(f'{width}x{height} -> {out_w}x{height}, {count} フレーム({count / fps:.1f}秒): {args.output}')


if __name__ == '__main__':
    main()
