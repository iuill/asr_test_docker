# reazonspeech-k2-v2_Docker

reazonspeech-k2-v2 を使用したリアルタイム音声文字起こしシステム

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

#### GPU版（推奨）

```bash
# リポジトリをクローン
git clone https://github.com/iuill/reazonspeech-k2-v2_Docker.git
cd reazonspeech-k2-v2_Docker

# Docker イメージをビルド
docker compose build

# 起動
docker compose up
```

#### CPU版

```bash
# CPU版でビルド・起動
docker compose --profile cpu up asr-cpu
```

### 使用方法

1. Docker コンテナを起動
2. ブラウザで `http://localhost:8000` にアクセス
3. マイクのアクセスを許可
4. 「開始」ボタンをクリックして文字起こし開始

### ローカル開発（Docker なし）

```bash
# 仮想環境を作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# PyTorch をインストール（CPU版）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# または GPU版（CUDA 11.8）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 依存関係をインストール
pip install -e .

# サーバーを起動
python -m src.main --device cpu  # または --device cuda
```

## 技術スタック

- **音声認識**: [reazonspeech-k2-v2](https://huggingface.co/reazon-research/reazonspeech-k2-v2) + [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- **VAD**: Silero VAD
- **サーバー**: FastAPI + WebSocket
- **フロントエンド**: HTML/JavaScript
- **コンテナ**: Docker

## プロジェクト構成

```
reazonspeech-k2-v2_Docker/
├── README.md                # このファイル
├── Dockerfile               # GPU版 Docker イメージ
├── Dockerfile.cpu           # CPU版 Docker イメージ
├── docker-compose.yml       # Docker Compose 設定
├── pyproject.toml           # Python プロジェクト設定
├── requirements.txt         # 依存関係
└── src/
    ├── __init__.py
    ├── main.py              # エントリポイント
    ├── server.py            # FastAPI WebSocket サーバー
    ├── transcription_engine.py  # sherpa-onnx ラッパー
    ├── audio_processor.py   # オーディオ処理
    ├── vad.py               # Silero VAD
    └── web/
        ├── index.html       # Web UI
        └── app.js           # クライアント JavaScript
```

## ライセンス

MIT License

## 参考

- [ReazonSpeech 公式](https://research.reazon.jp/projects/ReazonSpeech/)
- [sherpa-onnx GitHub](https://github.com/k2-fsa/sherpa-onnx)
- [iuill/WhisperLiveKit](https://github.com/iuill/WhisperLiveKit)（ベースプロジェクト）
