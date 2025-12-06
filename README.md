# ReazonSpeech Docker

ReazonSpeech モデルを使用したリアルタイム音声文字起こしシステム

## 対応モデル

| モデル | アーキテクチャ | パラメータ数 | 状態 |
|--------|---------------|-------------|------|
| [reazonspeech-k2-v2](https://huggingface.co/reazon-research/reazonspeech-k2-v2) | sherpa-onnx (Transducer) | 159M | ✅ 実装済み |
| [reazonspeech-espnet-v2](https://huggingface.co/reazon-research/reazonspeech-espnet-v2) | ESPnet (Conformer-Transducer) | 119M | ✅ 実装済み |

## 要件定義

### 機能要件

| 項目 | 内容 |
|------|------|
| 音声入力 | Windows マイクからのリアルタイム入力 |
| 文字起こし | 発話後 1-2秒以内の擬似リアルタイム表示 |
| モデル | reazonspeech-k2-v2（ONNX形式、sherpa-onnx経由） |
| UI | Web UI（ブラウザベース） |
| 実行環境 | Win11 + WSL2 + Docker |

### 非機能要件

| 項目 | 内容 |
|------|------|
| GPU対応 | NVIDIA GPU優先、CPU/AMD/NPU も考慮 |
| 遅延 | 2秒以内目標 |
| 並列接続 | 複数クライアント対応 |

### 技術的制約

- reazonspeech-k2-v2 は**オフラインモデル**（約30秒上限）
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

### 使用方法

1. Docker コンテナを起動
2. ブラウザで `http://localhost:13780` にアクセス（k2-v2の場合）
3. マイクのアクセスを許可
4. 「開始」ボタンをクリックして文字起こし開始

| サービス | ポート | URL |
|---------|--------|-----|
| k2-v2 | 13780 | http://localhost:13780 |
| espnet-v2 | 13781 | http://localhost:13781 |

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

### 共通
- **コンテナ**: Docker

## プロジェクト構成

```
reazonspeech-k2-v2_Docker/
├── README.md                    # このファイル
├── docker-compose.yml           # Docker Compose 設定（全サービス統括）
├── LICENSE
└── services/
    ├── k2-v2/                   # reazonspeech-k2-v2 用
    │   ├── Dockerfile           # GPU版
    │   ├── Dockerfile.cpu       # CPU版
    │   ├── pyproject.toml
    │   └── src/
    │       ├── main.py          # エントリポイント
    │       ├── server.py        # FastAPI WebSocket サーバー
    │       ├── transcription_engine.py  # sherpa-onnx ラッパー
    │       ├── audio_processor.py
    │       ├── vad.py           # Silero VAD
    │       └── web/
    │           ├── index.html
    │           └── app.js
    └── espnet-v2/               # reazonspeech-espnet-v2 用
        ├── Dockerfile           # GPU版
        ├── Dockerfile.cpu       # CPU版
        ├── pyproject.toml
        └── src/
            ├── main.py          # エントリポイント
            ├── server.py        # FastAPI WebSocket サーバー
            ├── transcription_engine.py  # ESPnet ラッパー
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
- [iuill/WhisperLiveKit](https://github.com/iuill/WhisperLiveKit)（ベースプロジェクト）
