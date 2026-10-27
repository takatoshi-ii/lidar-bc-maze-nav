"""動画の最後の「ロボットが止まったまま」の部分を自動で削る

使い方 (Windows の Python で実行):
  python tools/trim_tail.py 入力.mp4 出力.mp4
  python tools/trim_tail.py 入力.mp4 出力.mp4 --keep 1.0

コマ同士の差分で動きを判定する。video_recorder が左上・左下に出す文字(経過時間など)は
止まっていても変わり続けるので、その範囲は判定から外す。
"""
import argparse

import cv2
import numpy as np

HUD_WIDTH = 440      # 左上・左下の文字表示の幅[px] (720px幅の動画のとき)
HUD_HEIGHT = 115     # 同じく高さ[px]
PIXEL_DIFF = 25      # これより明るさが変わった画素を「変化あり」とする
MIN_CHANGED = 30     # 変化した画素がこれより多ければ「動いている」とする(圧縮ノイズ対策)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--keep', type=float, default=1.0, help='最後に動いてから残す秒数')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f'動画を開けません: {args.input}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit(f'フレームがありません: {args.input}')

    h, w = frames[0].shape[:2]
    scale = w / 720
    mask = np.ones((h, w), dtype=bool)
    mask[:int(HUD_HEIGHT * scale), :int(HUD_WIDTH * scale)] = False
    mask[h - int(HUD_HEIGHT * scale):, :int(HUD_WIDTH * scale)] = False

    last_moving = 0
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        changed = (cv2.absdiff(gray, prev) > PIXEL_DIFF) & mask
        if int(changed.sum()) > MIN_CHANGED:
            last_moving = i
        prev = gray

    end = min(len(frames), last_moving + 1 + int(args.keep * fps))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames[:end]:
        writer.write(frame)
    writer.release()
    print(f'{len(frames) / fps:.1f}秒 -> {end / fps:.1f}秒 '
          f'(最後の{(len(frames) - end) / fps:.1f}秒を削除): {args.output}')


if __name__ == '__main__':
    main()
