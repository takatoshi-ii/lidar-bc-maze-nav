"""複数の動画を順番につなげて1本にする(必要なら倍速にし、「2倍速」の表示を入れる)

使い方 (Windows の Python で実行):
  python tools/concat_videos.py 出力.mp4 video1.mp4 video2.mp4 video3.mp4
  python tools/concat_videos.py 出力.mp4 video1.mp4 video2.mp4 video3.mp4 --speed 2

倍速はフレームを間引いて作る(fpsは元のまま)。課題の要件に合わせ、倍速のときは右下に表示を入れる。
"""
import argparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT = 'C:/Windows/Fonts/YuGothB.ttc'


def render_label(text, font_path, size):
    font = ImageFont.truetype(font_path, size=size)
    left, top, right, bottom = font.getbbox(text)
    pad = size // 3
    w, h = right - left + pad * 2, bottom - top + pad * 2
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 4, fill=(200, 60, 30, 230))
    draw.text((pad - left, pad - top), text, font=font, fill=(255, 255, 255, 255))
    label = np.array(img)
    return label[:, :, 2::-1].astype(np.float32), label[:, :, 3:4].astype(np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--speed', type=int, default=1, help='倍速(整数)。2なら2倍速')
    parser.add_argument('--font', default=DEFAULT_FONT)
    args = parser.parse_args()

    first = cv2.VideoCapture(args.inputs[0])
    fps = first.get(cv2.CAP_PROP_FPS)
    width = int(first.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first.release()
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    label = None
    if args.speed > 1:
        rgb, alpha = render_label(f'{args.speed}倍速', args.font, max(width // 26, 14))
        lh, lw = rgb.shape[:2]
        lx, ly = width - lw - 10, height - lh - 10
        label = (rgb, alpha, lx, ly, lw, lh)

    written = 0
    for path in args.inputs:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise SystemExit(f'動画を開けません: {path}')
        if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) != (width, height):
            raise SystemExit(f'動画の大きさが1本目と違います: {path}')
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % args.speed == 0:
                if label is not None:
                    rgb, alpha, lx, ly, lw, lh = label
                    roi = frame[ly:ly + lh, lx:lx + lw].astype(np.float32)
                    frame[ly:ly + lh, lx:lx + lw] = (rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
                writer.write(frame)
                written += 1
            index += 1
        cap.release()
        print(f'  {path}: {index} フレーム')

    writer.release()
    print(f'{written} フレーム({written / fps:.1f}秒)を書き出しました: {args.output}')


if __name__ == '__main__':
    main()
