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

共通Web UIから各ASRモデルを選択して使用するマイクロサービス構成です。

```
┌─────────────────────────────────────────────────────────────┐
│                      WebUI Service                          │
│                    (Port: 13800)                            │
│  ┌──────────────┐  ┌─────────────────────────────────────┐  │
│  │  HTML/JS/CSS │  │  FastAPI Server                     │  │
│  │  - モデル選択 │  │  - 静的ファイル配信                  │  │
│  │  - 録音UI    │  │  - WebSocket Proxy (/ws/asr)        │  │
│  │  - 可視化    │  │  - モデル一覧API (/api/models)       │  │
│  │  - 結果表示  │  │  - ヘルスチェック (/health)          │  │
│  └──────────────┘  └─────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │ WebSocket/HTTP
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   k2-v2       │   │   espnet-v2   │   │ espnet-v2-onnx│
│ (内部:8000)   │   │ (内部:8000)   │   │ (内部:8000)   │
│               │   │               │   │               │
│ - /ws/asr     │   │ - /ws/asr     │   │ - /ws/asr     │
│ - /health     │   │ - /health     │   │ - /health     │
│ - /info       │   │ - /info       │   │ - /info       │
└───────────────┘   └───────────────┘   └───────────────┘
```

### 音声処理フロー

```
[Browser] <--WebSocket--> [WebUI Proxy] <--WebSocket--> [ASR Model]
   |                                                         |
   v                                                         v
[Microphone] --> [Audio Capture] --> [VAD] --> [Chunked Audio]
                                                     |
                                                     v
                                        [Model Inference (k2-v2/espnet)]
                                                     |
                                                     v
                                              [Transcription Result]
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

全サービスを起動（推奨）:

```bash
# GPU版（推奨）
docker compose up

# CPU版
docker compose --profile cpu up
```

個別のモデルサービスのみ起動:

```bash
# 特定のモデルのみ起動
docker compose up webui k2-v2

# CPU版で特定のモデルのみ
docker compose --profile cpu up webui-cpu k2-v2-cpu
```

> **Note**: `espnet-v2-onnx` の初回起動時にESPnetモデルをONNX形式にエクスポートするため、数分かかります。2回目以降はキャッシュされたONNXモデルが使用されます。

### 使用方法

1. Docker コンテナを起動
2. ブラウザで `http://localhost:13800` にアクセス
3. 使用したいモデルを選択
4. マイクのアクセスを許可
5. 「開始」ボタンをクリックして文字起こし開始

| サービス | ポート | URL | 説明 |
|---------|--------|-----|------|
| webui | 13800 | http://localhost:13800 | 共通Web UI（モデル選択可能） |

> **Note**: 各ASRモデル（k2-v2, espnet-v2, espnet-v2-onnx）は内部ネットワークでのみ動作し、Web UIからプロキシ経由でアクセスされます。

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
    ├── webui/                   # 共通 Web UI（フロントエンド）
    │   ├── Dockerfile
    │   ├── pyproject.toml
    │   └── src/
    │       ├── main.py          # エントリポイント
    │       ├── server.py        # WebSocket プロキシサーバー
    │       ├── config.py        # モデル設定
    │       └── web/
    │           ├── index.html   # モデル選択UI
    │           └── app.js       # 動的モデル切り替え
    ├── k2-v2/                   # reazonspeech-k2-v2 用（バックエンド）
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
    │       └── vad.py           # Silero VAD
    ├── espnet-v2/               # reazonspeech-espnet-v2 用（バックエンド）
    │   ├── .dockerignore
    │   ├── Dockerfile           # GPU版（ベースイメージ使用）
    │   ├── Dockerfile.cpu       # CPU版（ベースイメージ使用）
    │   ├── pyproject.toml
    │   └── src/
    │       ├── main.py          # エントリポイント
    │       ├── server.py        # FastAPI WebSocket サーバー
    │       ├── transcription_engine.py  # ESPnet ラッパー
    │       ├── audio_processor.py
    │       └── vad.py           # Silero VAD
    └── espnet-v2-onnx/          # reazonspeech-espnet-v2 ONNX版（バックエンド）
        ├── Dockerfile           # GPU版（ベースイメージ使用）
        ├── Dockerfile.cpu       # CPU版（ベースイメージ使用）
        ├── pyproject.toml
        └── src/
            ├── main.py          # エントリポイント
            ├── server.py        # FastAPI WebSocket サーバー
            ├── transcription_engine.py  # espnet_onnx ラッパー
            ├── audio_processor.py
            └── vad.py           # Silero VAD
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
