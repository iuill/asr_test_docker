# Azure Container Apps デプロイ (azd)

Azure Developer CLI (azd) を使用してASRサービスをAzure Container Appsにデプロイします。

## 前提条件

### Azure Developer CLI のインストール

```powershell
# winget でインストール
winget install Microsoft.Azd

# または PowerShell
powershell -ex AllSigned -c "Invoke-RestMethod 'https://aka.ms/install-azd.ps1' | Invoke-Expression"
```

### その他の要件
- Azure サブスクリプション
- 各クラウドAPIの認証情報

## クイックスタート

### 1. Azureにログイン

```powershell
azd auth login
```

### 2. 環境の初期化

```powershell
azd init
```

環境名を入力（例: `prod`, `dev`）

### 3. シークレットの設定

```powershell
# 必須シークレット
azd env set JWT_SECRET_KEY "your-jwt-secret-key"
azd env set AUTH_USERNAME "admin"
azd env set AUTH_PASSWORD "your-secure-password"
azd env set AZURE_SPEECH_KEY "your-azure-speech-key"
azd env set OPENAI_API_KEY "sk-your-openai-key"

# Google認証情報 (Base64エンコード)
# PowerShellでBase64エンコード:
# $base64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("google-stt-credentials.json"))
azd env set GOOGLE_CREDENTIALS_BASE64 "base64-encoded-json"

# オプション
azd env set AZURE_SPEECH_REGION "japaneast"
azd env set AUTH_ENABLED "true"
azd env set MIN_REPLICAS "0"
azd env set MAX_REPLICAS "1"
```

### 4. デプロイ

```powershell
# プロビジョニング + デプロイ (1コマンド)
azd up
```

## よく使うコマンド

| コマンド | 説明 |
|---------|------|
| `azd up` | プロビジョニング + デプロイ |
| `azd deploy` | アプリのみ再デプロイ |
| `azd provision` | インフラのみプロビジョニング |
| `azd down` | リソース削除 |
| `azd env list` | 環境一覧 |
| `azd env select <name>` | 環境切り替え |
| `azd monitor` | Azure Monitor を開く |

## ディレクトリ構成

```
/
├── azure.yaml                    # azd プロジェクト定義
├── infra/
│   ├── main.bicep               # メインテンプレート
│   ├── main.parameters.json     # パラメータ定義
│   └── modules/
│       ├── identity.bicep       # Managed Identity
│       ├── acr.bicep            # Container Registry
│       ├── keyvault.bicep       # Key Vault
│       ├── loganalytics.bicep   # Log Analytics
│       ├── environment.bicep    # Container Apps Environment
│       └── apps/
│           ├── webui.bicep      # WebUI (外部公開)
│           └── backend-service.bicep  # バックエンド (内部)
└── .azure/                       # 環境設定 (自動生成)
```

## デプロイされるリソース

| リソース | 説明 |
|---------|------|
| Resource Group | `rg-{環境名}` |
| Container Registry | イメージ保管 |
| Key Vault | シークレット管理 |
| Log Analytics | ログ収集 |
| Container Apps Environment | 実行環境 |
| Container Apps (8個) | サービス |

## 環境変数一覧

| 変数名 | 必須 | 説明 | デフォルト |
|--------|------|------|-----------|
| `JWT_SECRET_KEY` | ✅ | JWT署名キー | - |
| `AUTH_USERNAME` | ✅ | ログインユーザー名 | - |
| `AUTH_PASSWORD` | ✅ | ログインパスワード | - |
| `AZURE_SPEECH_KEY` | ✅ | Azure Speech APIキー | - |
| `OPENAI_API_KEY` | ✅ | OpenAI APIキー | - |
| `GOOGLE_CREDENTIALS_BASE64` | ✅ | Google認証情報 (Base64) | - |
| `AZURE_SPEECH_REGION` | - | Azure Speechリージョン | `japaneast` |
| `AUTH_ENABLED` | - | 認証有効化 | `true` |
| `MIN_REPLICAS` | - | 最小レプリカ数 | `0` |
| `MAX_REPLICAS` | - | 最大レプリカ数 | `3` |
| `CUSTOM_DOMAIN` | - | カスタムドメイン | - |

## トラブルシューティング

### デプロイ状況の確認

```powershell
azd show
```

### ログの確認

```powershell
# Azure Monitor を開く
azd monitor

# CLI でログ確認
az containerapp logs show --name <app-name> --resource-group <rg-name> --follow
```

### 環境のリセット

```powershell
# リソース削除
azd down

# 環境設定をクリア
azd env refresh
```

## リソースの削除

```powershell
# 確認プロンプトあり
azd down

# 確認なしで削除
azd down --force
```
