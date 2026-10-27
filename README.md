# lidar-bc-maze-nav

人がキーボードで操作した「お手本走行」から、AIが運転を真似して学ぶ **模倣学習（Behavior Cloning）** で、
差動二輪ロボットに迷路を自律走行させるプロジェクトです。
AIはロボットに付いた **LiDAR（周囲360°の距離センサ）の値だけ** を見て、「前進・左旋回・右旋回」を判断します。

Physical AI 応用編1 講座の最終課題として作成しました。

- 動画: https://youtu.be/QWV9pXOS_lI
- 環境: ROS 2 Humble / Ignition Gazebo Fortress / Python / PyTorch（CPU）

## ベースにしたサンプルと、追加した部分

このリポジトリは [OsawaKousei/2DRoboPrac_ws](https://github.com/OsawaKousei/2DRoboPrac_ws) の fork です。
最終課題サンプル「ROS2とGazeboの連携によるロボット操作」（[Qiita記事](https://qiita.com/N622/items/9d89f77d85d9da0af29e)）をベースにしています。

| 区分 | 内容 |
|---|---|
| 元のサンプルから使用 | ロボットモデル（`LidarRobo`）、迷路フィールド、ROS 2 と Gazebo のブリッジ設定（`ros2_ws/src/nav_dev`）、キー入力パッケージ（`key_event_*`） |
| **今回作成** | `ros2_ws/src/bc_nav`（下記）、`tools/`（動画編集スクリプト） |
| 修正 | `key_event_msgs/package.xml` の依存名のスペルミス（`generaters` → `generators`） |

## 仕組み

```
① お手本走行           ② 学習                     ③ 自律走行
人がテレオペ    →   LiDAR → 操作 を        →   AIがLiDARから操作を推論
(bc_teleop)         ニューラルネットで学習        (bc_driver)
LiDARと操作を記録    (train_bc)                  引っかかったら自動でリカバリー
(data_recorder)
```

### `bc_nav` パッケージ

| ファイル | 役割 |
|---|---|
| `bc_teleop.py` | お手本用のテレオペ。自動で前進し、`a` / `d` を押している間だけ曲がる（レースゲーム風） |
| `data_recorder.py` | `/scan` と `/cmd_vel` をペアで約10Hz記録し、`~/bc_nav_data/session_*.npz` に保存 |
| `scan_utils.py` | LiDAR 360点を72方向の最小距離に縮約・正規化。行動3クラスと速度の定義 |
| `train_bc.py` | PyTorch の MLP（72→128→64→3）で分類学習。左右反転によるデータ拡張、クラス重み付け |
| `bc_driver.py` | 学習済みモデルで自律走行。判断の平滑化、引っかかり検出とリカバリー、評価結果のCSV保存 |
| `video_recorder.py` | 真上カメラの映像に、AIの判断・リカバリー表示・走行軌跡・壁を重ねてMP4保存 |
| `launch/bc_sim.launch.py` | `mode:=record`（お手本収集）/ `mode:=drive`（自律走行）を切り替えて起動 |
| `worlds/bc_world.sdf` | 真上視点、影のノイズ対策、撮影用の真上カメラを追加したワールド |

### 工夫した点

- **操作しやすいお手本用テレオペ**：元のテレオペは速く、キーを離しても止まらないため運転が難しかった。自動前進＋押している間だけ旋回にし、学習する3クラスとも一対一に対応させた
- **左右反転によるデータ拡張**：スキャンを左右反転し、左右のラベルを入れ替えてデータを2倍にした
- **クラス重み付けの調整**：前進が多いデータに対して重みを付けすぎると「曲がりすぎ」になったため、重みを平方根で弱めた
- **AI＋ルールの組み合わせ**：お手本に「バック」が無いため、AIだけでは壁に引っかかると抜け出せない。進めない状態を検出したらバックして開けた方向へ向き直すリカバリーを追加した
- **汎化性能の評価**：迷路は中央の壁で区画A・Bに分かれている。お手本は区画Aだけで集め、一度も走っていない区画Bでも走れるかを確認した

## 結果（各90秒の自律走行）

| 条件 | 走行距離 | リカバリー回数 |
|---|---|---|
| ① 区画A：学習したルート | 18.3 m | 0 回 |
| ② 区画A：スタート位置を変更 | 17.7 m | 1 回 |
| ③ 区画B：未学習エリア | 16.5 m | 2 回 |

## 使い方

ROS 2 Humble と Ignition Gazebo Fortress が入った Ubuntu 22.04 で動かします。

```bash
cd ros2_ws
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
sudo apt install -y xterm python3-opencv
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-up-to bc_nav
source install/setup.bash
```

```bash
# 1. お手本走行を記録（xterm のウィンドウで w で開始、a/d で旋回、スペースで一時停止）
ros2 launch bc_nav bc_sim.launch.py mode:=record

# 2. 学習（~/bc_nav_data/model.pt に保存）
ros2 run bc_nav train_bc

# 3. 自律走行（90秒で自動停止し、評価結果を ~/bc_nav_data/eval_results.csv に追記）
ros2 launch bc_nav bc_sim.launch.py mode:=drive duration:=90

# 区画B（未学習エリア）から走らせる / 動画も保存する
ros2 launch bc_nav bc_sim.launch.py mode:=drive duration:=90 x:=4.68 y:=1.06 record_video:=true
```

`tools/` の動画編集スクリプトは Windows の Python（OpenCV, Pillow）で動かします。

| スクリプト | 役割 |
|---|---|
| `add_caption.py` | 動画の最初の数秒に日本語テロップを入れる |
| `trim_tail.py` | 最後のロボットが止まっている部分を自動で削る |
| `concat_videos.py` | 複数の動画をつなげる（倍速と「2倍速」表示にも対応） |

## ライセンス

元のリポジトリと同じ Apache License 2.0 です（`LICENSE`）。

---

## 元のリポジトリの README

> # About
> This is a repositry for developing 2D auto drive robot.
> Description is in following article.
> https://qiita.com/N622/items/6ca879f8b1f9d667e561
