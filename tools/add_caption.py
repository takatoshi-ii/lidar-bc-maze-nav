"""動画の中央に日本語のテロップを入れる(最初の数秒だけ表示し、最後はフェードアウト)

使い方 (Windows の Python で実行):
  python tools/add_caption.py 入力.mp4 出力.mp4 "①学習したルートを走行"
  python tools/add_caption.py 入力.mp4 出力.mp4 "①学習したルートを走行" --seconds 10

OpenCV は日本語を描けないので、文字は Pillow で描く。
"""
import argparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT = 'C:/Windows/Fonts/YuGothB.ttc'
FADE_SECONDS = 0.5


def render_caption(text, frame_width, font_path, size=None, bg=(30, 90, 200)):
    """テロップ(色付きの帯+白文字)をRGBA画像として作る。

    size を指定すればその文字サイズで固定。省略時は画面幅に収まるよう自動で縮める。
    """
    pad_x, pad_y = frame_width // 30, frame_width // 50
    fixed = size is not None
    size = size or max(frame_width // 16, 20)
    while True:
        font = ImageFont.truetype(font_path, size=size)
        left, top, right, bottom = font.getbbox(text)
        w, h = right - left + pad_x * 2, bottom - top + pad_y * 2
        if fixed or w <= frame_width * 0.95 or size <= 10:
            break
        size -= 2
    if w > frame_width:
        raise SystemExit(f'文字サイズ {size}px ではテロップ(幅{w}px)が動画の幅{frame_width}pxに収まりません')
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 4, fill=(*bg, 230))
    draw.text((pad_x - left, pad_y - top), text, font=font, fill=(255, 255, 255, 255))
    return np.array(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('text')
    parser.add_argument('--seconds', type=float, default=10.0, help='テロップを表示する秒数')
    parser.add_argument('--size', type=int, help='文字サイズ[px]。複数の動画で揃えたいときに指定する')
    parser.add_argument('--bottom', type=int,
                        help='テロップの下端のy座標[px]。省略時は画面中央。'
                             '区画Bの最下部(中央の壁の上端)に揃えるなら 356、'
                             '区画Bの下の通路(壁と壁の間)の中央に置くなら 333')
    parser.add_argument('--bg', default='#1E5AC8', help='帯の色(例: #1E5AC8 = 青)')
    parser.add_argument('--font', default=DEFAULT_FONT)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f'動画を開けません: {args.input}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    bg = tuple(int(args.bg.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
    caption = render_caption(args.text, width, args.font, args.size, bg)
    ch, cw = caption.shape[:2]
    x0 = (width - cw) // 2
    y0 = (height - ch) // 2 if args.bottom is None else max(args.bottom - ch, 0)
    rgb = caption[:, :, 2::-1].astype(np.float32)   # RGBA -> BGR
    alpha = caption[:, :, 3:4].astype(np.float32) / 255.0

    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_no / fps
        if t < args.seconds:
            fade = min(1.0, (args.seconds - t) / FADE_SECONDS)
            roi = frame[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
            a = alpha * fade
            frame[y0:y0 + ch, x0:x0 + cw] = (rgb * a + roi * (1 - a)).astype(np.uint8)
        writer.write(frame)
        frame_no += 1

    cap.release()
    writer.release()
    print(f'{frame_no} フレーム({frame_no / fps:.1f}秒)を書き出しました: {args.output}')


if __name__ == '__main__':
    main()
