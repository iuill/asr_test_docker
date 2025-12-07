# Google Speech-to-Text ストリーミング仕様

## 概要

Google Cloud Speech-to-Text APIのストリーミング認識では、1つのレスポンスに**複数の認識結果（results配列）**が含まれることがある。これは連続する複数の発話区間を同時に処理しているためである。

## APIレスポンス構造

```
StreamingRecognizeResponse
└── results[] (StreamingRecognitionResult の配列)
    ├── results[0]: 確定に近い発話区間
    ├── results[1]: まだ不安定な次の発話区間
    └── ...
```

### StreamingRecognitionResult の主要フィールド

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `alternatives[]` | array | 認識候補（信頼度順） |
| `is_final` | bool | 最終結果かどうか |
| `stability` | float | 暫定結果の安定度（0.0〜1.0） |
| `result_end_time` | Duration | 音声開始からの経過時間 |
| `channel_tag` | int | マルチチャンネル音声用 |
| `language_code` | string | 検出された言語コード |

### SpeechRecognitionAlternative の主要フィールド

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `transcript` | string | 認識テキスト |
| `confidence` | float | 信頼度（is_final=true時のみ） |
| `words[]` | array | 単語レベル情報（enable_word_time_offsets=true時） |

## 暫定結果の特性

### stability（安定度）

- **0.0〜1.0** の範囲
- 高いほど変更される可能性が低い
- 典型的な値:
  - `0.9`: かなり安定（result_index=0 でよく見られる）
  - `0.01`: 非常に不安定（result_index=1以降でよく見られる）

### result_index（結果インデックス）

1つのレスポンスに複数のresultsが含まれる場合：

| result_index | 特徴 |
|--------------|------|
| 0 | 確定に近い発話。stability が高い（0.9程度） |
| 1以降 | まだ認識中の次の発話。stability が低い（0.01程度） |

## フロントエンド実装

### フィルタリング戦略

暫定結果の表示チラつきを防ぐため、以下の条件でフィルタリング：

```javascript
// Google STT の暫定結果フィルタリング
if (modelId === 'google-stt') {
    const resultIndex = data.provider_info?.result_index ?? 0;

    // result_index=0 の場合のみ表示
    // result_index=0 は常に高い stability (~0.9) を持つ
    if (resultIndex === 0) {
        conn.partialText = data.text;
    }
}
```

### 理由

- **result_index=0 のみ**: 確定に近い発話のみを表示し、まだ認識中の次の発話（result_index > 0）は無視
- **stability チェックは不要**: result_index=0 は常に高い stability（約0.9）を持つため

## WebSocket メッセージ形式

### サーバー → クライアント

```json
{
    "type": "transcription",
    "text": "認識されたテキスト",
    "start_time": 0.0,
    "end_time": 15.62,
    "is_final": false,
    "provider_info": {
        "stability": 0.9,
        "confidence": 0.0,
        "result_index": 0
    }
}
```

### provider_info フィールド

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `stability` | float | 暫定結果の安定度 |
| `confidence` | float | 認識信頼度（is_final=true時のみ有効） |
| `result_index` | int | results配列内のインデックス |

## 参考リンク

- [Google Cloud Speech-to-Text RPC Reference](https://docs.cloud.google.com/speech-to-text/docs/reference/rpc/google.cloud.speech.v1)
- [Transcribe audio from streaming input](https://docs.cloud.google.com/speech-to-text/docs/v1/transcribe-streaming-audio)
