"""記録したデータからLiDAR -> 行動の分類器(MLP)を学習する

使い方:
  ros2 run bc_nav train_bc --data-dir ~/bc_nav_data --out ~/bc_nav_data/model.pt
"""
import argparse
import glob
import os

import numpy as np
import torch
from torch import nn

from bc_nav.scan_utils import ACTIONS, DEFAULT_CLIP, mirror_bins, mirror_label


def build_model(n_bins, n_actions=len(ACTIONS)):
    return nn.Sequential(
        nn.Linear(n_bins, 128), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, n_actions),
    )


def load_sessions(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, 'session_*.npz')))
    if not files:
        raise SystemExit(f'{data_dir} に session_*.npz がありません')
    scans, labels = [], []
    for f in files:
        d = np.load(f)
        scans.append(d['scans'])
        labels.append(d['labels'])
        print(f'  {os.path.basename(f)}: {len(d["labels"])} サンプル')
    return np.concatenate(scans), np.concatenate(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=os.path.expanduser('~/bc_nav_data'))
    parser.add_argument('--out', default=os.path.expanduser('~/bc_nav_data/model.pt'))
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--no-mirror', action='store_true', help='左右反転のデータ拡張を使わない')
    args = parser.parse_args()

    x, y = load_sessions(args.data_dir)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    n_val = max(len(y) // 5, 1)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    if not args.no_mirror:
        x_train = np.concatenate([x_train, mirror_bins(x_train)])
        y_train = np.concatenate([y_train, mirror_label(y_train)])

    counts = np.bincount(y_train, minlength=len(ACTIONS))
    print('学習データ:', ', '.join(f'{a}={c}' for a, c in zip(ACTIONS, counts)), f'/ 検証 {len(y_val)}')

    # 前進ばかりになりがちなので、少ないクラスを重く扱う
    # (比率そのままだと曲がりすぎるので平方根で弱める)
    weights = torch.tensor(np.sqrt(counts.sum() / np.maximum(counts, 1) / len(ACTIONS)),
                           dtype=torch.float32)

    model = build_model(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    xt, yt = torch.from_numpy(x_train), torch.from_numpy(y_train).long()
    xv, yv = torch.from_numpy(x_val), torch.from_numpy(y_val).long()

    best_acc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(yt), 128):
            b = perm[i:i + 128]
            optimizer.zero_grad()
            loss = loss_fn(model(xt[b]), yt[b])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_acc = (model(xv).argmax(1) == yv).float().mean().item()
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f'epoch {epoch:3d}  loss {loss.item():.3f}  val_acc {val_acc:.3f}')

    model.load_state_dict(best_state)
    with torch.no_grad():
        pred = model(xv).argmax(1).numpy()
    print(f'\n最良の検証正解率: {best_acc:.3f}')
    print('混同行列 (行=お手本, 列=AI):', ACTIONS)
    conf = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=int)
    for t, p in zip(y_val, pred):
        conf[t, p] += 1
    print(conf)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({'state_dict': best_state, 'n_bins': x.shape[1], 'clip': DEFAULT_CLIP}, args.out)
    print(f'モデルを保存: {args.out}')


if __name__ == '__main__':
    main()
