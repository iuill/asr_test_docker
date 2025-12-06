# ReazonSpeech Docker

ReazonSpeech モデルを使用したリアルタイム音声文字起こしシステム

## 対応モデル

| モデル | アーキテクチャ | パラメータ数 | 速度 |
|--------|---------------|-------------|------|
| [reazonspeech-k2-v2](https://huggingface.co/reazon-research/reazonspeech-k2-v2) | sherpa-onnx (Transducer) | 159M | 高速 |
| [reazonspeech-espnet-v2](https://huggingface.co/reazon-research/reazonspeech-espnet-v2) | ESPnet (Conformer-Transducer) | 119M | 標準 |
| reazonspeech-espnet-v2-onnx | ESPnet ONNX (Conformer-Transducer) | 119M | 高速 |

> **Note**: `espnet-v2-onnx` は `espnet-v2` と同じモデルをONNX形式に変換して使用するため、精度は同等で推論速度が向上します。

## 要件定義

### 機能要件

| 項目 | 内容 |
|------|------|
| 音声入力 | Windows マイクからのリアルタイム入力 |
| 文字起こし | 発話後 1-2秒以内の擬似リアルタイム表示 |
| モデル | reazonspeech-k2-v2 / reazonspeech-espnet-v2 / reazonspeech-espnet-v2-onnx |
| UI | Web UI（ブラウザベース） |
| 実行環境 | Win11 + WSL2 + Docker |

### 非機能要件

| 項目 | 内容 |
|------|------|
| GPU対応 | NVIDIA GPU優先、CPU/AMD/NPU も考慮 |
| 遅延 | 2秒以内目標 |
| 並列接続 | 複数クライアント対応 |

### 技術的制約

- 各モデルは**オフラインモデル**（約30秒上限）
- リアルタイム認識には VAD + チャンク分割による擬似ストリーミングで実現

## アーキテクチャ

```
[Browser] <--WebSocket--> [Docker Container]
   |                            |
   v                            v
[Microphone] --> [Audio Capture] --> [VAD] --> [Chunked Audio]
                                                     |
                                                     v
                                        [sherpa-onnx + reazonspeech-k2-v2]
                                                     |
                                                     v
                                              [Transcription Result]
                                                     |
                                                     v
                                              [WebSocket Response]
```

## セットアップ

### 必要環境

- Windows 11
- WSL2
- Docker Desktop（WSL2バックエンド）
- NVIDIA GPU + NVIDIA Container Toolkit（GPU使用時）

### インストール

```bash
# リポジトリをクローン
git clone https://github.com/iuill/reazonspeech-k2-v2_Docker.git
cd reazonspeech-k2-v2_Docker
```

### ビルド

本プロジェクトは3層のベースイメージ構造を採用しており、2回目以降のビルドが高速化されます。

```
Layer 1: common-base           ← PyTorch + 共通依存（最も時間がかかる）
Layer 2: k2-v2-base            ← + sherpa-onnx
         espnet-v2-base        ← + espnet
         espnet-v2-onnx-base   ← + espnet_onnx
Layer 3: アプリイメージ        ← ソースコードのみ（高速）
```

#### 初回ビルド（ベースイメージの作成）

```powershell
# Windows (PowerShell)
.\scripts\build-base-images.ps1 -Target gpu   # GPU版
.\scripts\build-base-images.ps1 -Target cpu   # CPU版
.\scripts\build-base-images.ps1 -Target all   # 両方
```

```bash
# Linux / WSL
./scripts/build-base-images.sh gpu   # GPU版
./scripts/build-base-images.sh cpu   # CPU版
./scripts/build-base-images.sh all   # 両方
```

#### アプリイメージのビルド

```bash
# ベースイメージ作成後に実行
docker compose build
```

> **Note**: ソースコード変更時は `docker compose build` のみで高速にリビルドできます。
> 依存関係を変更した場合は、ベースイメージの再ビルドが必要です。

### 起動

#### reazonspeech-k2-v2

```bash
# GPU版（推奨）
docker compose up k2-v2

# CPU版
docker compose --profile cpu up k2-v2-cpu
```

#### reazonspeech-espnet-v2

```bash
# GPU版（推奨）
docker compose up espnet-v2

# CPU版
docker compose --profile cpu up espnet-v2-cpu
```

#### reazonspeech-espnet-v2-onnx（高速版）

```bash
# GPU版（推奨）
docker compose up espnet-v2-onnx

# CPU版
docker compose --profile cpu up espnet-v2-onnx-cpu
```

> **Note**: 初回起動時にESPnetモデルをONNX形式にエクスポートするため、数分かかります。2回目以降はキャッシュされたONNXモデルが使用されます。

### 使用方法

1. Docker コンテナを起動
2. ブラウザで `http://localhost:13780` にアクセス（k2-v2の場合）
3. マイクのアクセスを許可
4. 「開始」ボタンをクリックして文字起こし開始

| サービス | ポート | URL |
|---------|--------|-----|
| k2-v2 | 13780 | http://localhost:13780 |
| espnet-v2 | 13781 | http://localhost:13781 |
| espnet-v2-onnx | 13782 | http://localhost:13782 |

### ローカル開発（Docker なし）

```bash
# 仮想環境を作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# PyTorch をインストール（CPU版）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# または GPU版（CUDA 11.8）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# k2-v2 の依存関係をインストール
cd services/k2-v2
pip install -e .

# サーバーを起動
python -m src.main --device cpu  # または --device cuda
```

## 技術スタック

### k2-v2
- **音声認識**: [reazonspeech-k2-v2](https://huggingface.co/reazon-research/reazonspeech-k2-v2) + [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- **VAD**: Silero VAD
- **サーバー**: FastAPI + WebSocket
- **フロントエンド**: HTML/JavaScript

### espnet-v2
- **音声認識**: [reazonspeech-espnet-v2](https://huggingface.co/reazon-research/reazonspeech-espnet-v2) + [ESPnet](https://github.com/espnet/espnet)
- **VAD**: Silero VAD
- **サーバー**: FastAPI + WebSocket
- **フロントエンド**: HTML/JavaScript

### espnet-v2-onnx
- **音声認識**: [reazonspeech-espnet-v2](https://huggingface.co/reazon-research/reazonspeech-espnet-v2) + [espnet_onnx](https://github.com/espnet/espnet_onnx)
- **VAD**: Silero VAD
- **サーバー**: FastAPI + WebSocket
- **フロントエンド**: HTML/JavaScript
- **特徴**: ESPnetモデルをONNX形式に変換し、ONNX Runtimeで高速推論

### 共通
- **コンテナ**: Docker

## プロジェクト構成

```
reazonspeech-k2-v2_Docker/
├── README.md                    # このファイル
├── docker-compose.yml           # Docker Compose 設定（全サービス統括）
├── LICENSE
├── scripts/
│   ├── build-base-images.ps1    # ベースイメージビルド（Windows）
│   └── build-base-images.sh     # ベースイメージビルド（Linux/WSL）
└── services/
    ├── base/                    # ベースイメージ定義
    │   ├── Dockerfile.common-gpu     # 共通ベース（PyTorch + CUDA）
    │   ├── Dockerfile.common-cpu     # 共通ベース（PyTorch CPU）
    │   ├── Dockerfile.k2-v2-gpu      # k2-v2用（+ sherpa-onnx）
    │   ├── Dockerfile.k2-v2-cpu
    │   ├── Dockerfile.espnet-v2-gpu  # espnet用（+ espnet）
    │   ├── Dockerfile.espnet-v2-cpu
    │   ├── Dockerfile.espnet-v2-onnx-gpu  # espnet-onnx用（+ espnet_onnx）
    │   └── Dockerfile.espnet-v2-onnx-cpu
    ├── k2-v2/                   # reazonspeech-k2-v2 用
    │   ├── .dockerignore
    │   ├── Dockerfile           # GPU版（ベースイメージ使用）
    │   ├── Dockerfile.cpu       # CPU版（ベースイメージ使用）
    │   ├── pyproject.toml
    │   ├── requirements.txt
    │   └── src/
    │       ├── main.py          # エントリポイント
    │       ├── server.py        # FastAPI WebSocket サーバー
    │       ├── transcription_engine.py  # sherpa-onnx ラッパー
    │       ├── audio_processor.py
    │       ├── vad.py           # Silero VAD
    │       └── web/
    │           ├── index.html
    │           └── app.js
    ├── espnet-v2/               # reazonspeech-espnet-v2 用
    │   ├── .dockerignore
    │   ├── Dockerfile           # GPU版（ベースイメージ使用）
    │   ├── Dockerfile.cpu       # CPU版（ベースイメージ使用）
    │   ├── pyproject.toml
    │   └── src/
    │       ├── main.py          # エントリポイント
    │       ├── server.py        # FastAPI WebSocket サーバー
    │       ├── transcription_engine.py  # ESPnet ラッパー
    │       ├── audio_processor.py
    │       ├── vad.py           # Silero VAD
    │       └── web/
    │           ├── index.html
    │           └── app.js
    └── espnet-v2-onnx/          # reazonspeech-espnet-v2 ONNX版（高速）
        ├── Dockerfile           # GPU版（ベースイメージ使用）
        ├── Dockerfile.cpu       # CPU版（ベースイメージ使用）
        ├── pyproject.toml
        └── src/
            ├── main.py          # エントリポイント
            ├── server.py        # FastAPI WebSocket サーバー
            ├── transcription_engine.py  # espnet_onnx ラッパー
            ├── audio_processor.py
            ├── vad.py           # Silero VAD
            └── web/
                ├── index.html
                └── app.js
```

## ライセンス

Apache License 2.0

## 参考

- [ReazonSpeech 公式](https://research.reazon.jp/projects/ReazonSpeech/)
- [ReazonSpeech GitHub](https://github.com/reazon-research/reazonspeech)
- [sherpa-onnx GitHub](https://github.com/k2-fsa/sherpa-onnx)
- [ESPnet GitHub](https://github.com/espnet/espnet)
- [espnet_onnx GitHub](https://github.com/espnet/espnet_onnx)
- [iuill/WhisperLiveKit](https://github.com/iuill/WhisperLiveKit)（ベースプロジェクト）
