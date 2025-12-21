// =============================================================================
// ASR Service - Azure Container Apps Infrastructure (azd対応)
// =============================================================================
// このテンプレートは以下のリソースをデプロイします:
// - Azure Container Registry (ACR)
// - Azure Key Vault
// - User-Assigned Managed Identity
// - Log Analytics Workspace
// - Container Apps Environment
// - 8つのContainer Apps (webui + 7つのSTTサービス)
// =============================================================================

targetScope = 'subscription'

// -----------------------------------------------------------------------------
// Parameters (azd が自動で渡すもの)
// -----------------------------------------------------------------------------

@minLength(1)
@maxLength(64)
@description('環境名 (azd env name)')
param environmentName string

@minLength(1)
@description('デプロイ先のAzureリージョン')
param location string

@description('リソースのプレフィックス名')
param prefix string = 'asr'

// -----------------------------------------------------------------------------
// シークレット (azd env set で設定)
// -----------------------------------------------------------------------------

@secure()
@description('JWT署名キー')
param jwtSecretKey string

@secure()
@description('認証ユーザー名')
param authUsername string

@secure()
@description('認証パスワード')
param authPassword string

@secure()
@description('Azure Speech APIキー')
param azureSpeechKey string

@description('Azure Speechリージョン')
param azureSpeechRegion string = 'japaneast'

@secure()
@description('OpenAI APIキー')
param openaiApiKey string

@secure()
@description('Google認証情報JSON (Base64エンコード)')
param googleCredentialsBase64 string

// -----------------------------------------------------------------------------
// オプション設定
// -----------------------------------------------------------------------------

@description('認証を有効にするか')
param authEnabled bool = true

@description('最小レプリカ数 (0でゼロスケール)')
param minReplicas int = 0

@description('最大レプリカ数')
param maxReplicas int = 3

@description('カスタムドメイン (空の場合はAzure提供ドメインを使用)')
param customDomain string = ''

// -----------------------------------------------------------------------------
// Variables
// -----------------------------------------------------------------------------

var resourceToken = toLower(take(uniqueString(subscription().id, environmentName, location), 6))
var resourceGroupName = 'rg-${environmentName}'
var acrName = '${prefix}${resourceToken}acr'

var tags = {
  'azd-env-name': environmentName
  Project: 'asr-service'
  ManagedBy: 'azd'
}

// サービス定義 (実際に存在するディレクトリに合わせる)
// 注意: shortName は Container App名に使用 (32文字制限)
// 注意: 全サービスが8000ポートで起動するため、portは8000に統一
var backendServices = [
  {
    name: 'azure-stt'
    shortName: 'az-stt'
    displayName: 'Azure STT'
    port: 8000
    envVars: [
      { name: 'AZURE_SPEECH_KEY', secretRef: 'azure-speech-key' }
      { name: 'AZURE_SPEECH_REGION', value: azureSpeechRegion }
    ]
  }
  {
    name: 'azure-stt-diarization'
    shortName: 'az-stt-diar'
    displayName: 'Azure STT Diarization'
    port: 8000
    envVars: [
      { name: 'AZURE_SPEECH_KEY', secretRef: 'azure-speech-key' }
      { name: 'AZURE_SPEECH_REGION', value: azureSpeechRegion }
    ]
  }
  {
    name: 'google-stt-v1'
    shortName: 'gcp-stt-v1'
    displayName: 'Google STT V1'
    port: 8000
    envVars: [
      { name: 'GOOGLE_APPLICATION_CREDENTIALS', value: '/secrets/google/credentials.json' }
      { name: 'GOOGLE_STT_MODEL', value: 'default' }
    ]
    useGoogleCredentials: true
  }
  {
    name: 'google-stt-chirp2'
    shortName: 'gcp-chirp2'
    displayName: 'Google STT Chirp2'
    port: 8000
    envVars: [
      { name: 'GOOGLE_APPLICATION_CREDENTIALS', value: '/secrets/google/credentials.json' }
      { name: 'GOOGLE_STT_MODEL', value: 'chirp_2' }
      { name: 'GOOGLE_STT_LOCATION', value: 'asia-southeast1' }
    ]
    useGoogleCredentials: true
  }
  {
    name: 'google-stt-chirp3'
    shortName: 'gcp-chirp3'
    displayName: 'Google STT Chirp3'
    port: 8000
    envVars: [
      { name: 'GOOGLE_APPLICATION_CREDENTIALS', value: '/secrets/google/credentials.json' }
      { name: 'GOOGLE_STT_MODEL', value: 'chirp_3' }
      { name: 'GOOGLE_STT_LOCATION', value: 'asia-south1' }
    ]
    useGoogleCredentials: true
  }
  {
    name: 'openai-stt'
    shortName: 'oai-stt'
    displayName: 'OpenAI STT'
    port: 8000
    envVars: [
      { name: 'OPENAI_API_KEY', secretRef: 'openai-api-key' }
    ]
  }
  {
    name: 'openai-stt-mini'
    shortName: 'oai-stt-mini'
    displayName: 'OpenAI STT Mini'
    port: 8000
    envVars: [
      { name: 'OPENAI_API_KEY', secretRef: 'openai-api-key' }
    ]
  }
]

// -----------------------------------------------------------------------------
// Resource Group
// -----------------------------------------------------------------------------

resource rg 'Microsoft.Resources/resourceGroups@2022-09-01' = {
  name: resourceGroupName
  location: location
  tags: tags
}

// -----------------------------------------------------------------------------
// Modules
// -----------------------------------------------------------------------------

// User-Assigned Managed Identity
module identity 'modules/identity.bicep' = {
  name: 'identity-deployment'
  scope: rg
  params: {
    name: '${prefix}-${resourceToken}-identity'
    location: location
    tags: tags
  }
}

// Azure Container Registry
module acr 'modules/acr.bicep' = {
  name: 'acr-deployment'
  scope: rg
  params: {
    name: acrName
    location: location
    tags: tags
    managedIdentityPrincipalId: identity.outputs.principalId
  }
}

// Azure Key Vault
module keyVault 'modules/keyvault.bicep' = {
  name: 'keyvault-deployment'
  scope: rg
  params: {
    name: '${prefix}-${resourceToken}-kv'
    location: location
    tags: tags
    managedIdentityPrincipalId: identity.outputs.principalId
    secrets: [
      { name: 'jwt-secret-key', value: jwtSecretKey }
      { name: 'auth-username', value: authUsername }
      { name: 'auth-password', value: authPassword }
      { name: 'azure-speech-key', value: azureSpeechKey }
      { name: 'openai-api-key', value: openaiApiKey }
      { name: 'google-credentials-base64', value: googleCredentialsBase64 }
    ]
  }
}

// Log Analytics Workspace
module logAnalytics 'modules/loganalytics.bicep' = {
  name: 'loganalytics-deployment'
  scope: rg
  params: {
    name: '${prefix}-${resourceToken}-logs'
    location: location
    tags: tags
  }
}

// Container Apps Environment
module containerAppsEnv 'modules/environment.bicep' = {
  name: 'environment-deployment'
  scope: rg
  params: {
    name: '${prefix}-${resourceToken}-env'
    location: location
    tags: tags
    logAnalyticsWorkspaceId: logAnalytics.outputs.workspaceId
    logAnalyticsSharedKey: logAnalytics.outputs.sharedKey
  }
}

// WebUI Container App
module webui 'modules/apps/webui.bicep' = {
  name: 'webui-deployment'
  scope: rg
  params: {
    name: '${prefix}-${resourceToken}-webui'
    location: location
    tags: tags
    environmentId: containerAppsEnv.outputs.environmentId
    managedIdentityId: identity.outputs.id
    acrLoginServer: acr.outputs.loginServer
    keyVaultName: keyVault.outputs.name
    authEnabled: authEnabled
    minReplicas: minReplicas
    maxReplicas: maxReplicas
    customDomain: customDomain
    backendServices: backendServices
  }
}

// Backend Services
module backendApps 'modules/apps/backend-service.bicep' = [for service in backendServices: {
  name: '${service.shortName}-deployment'
  scope: rg
  params: {
    name: '${prefix}-${resourceToken}-${service.shortName}'
    location: location
    tags: tags
    environmentId: containerAppsEnv.outputs.environmentId
    managedIdentityId: identity.outputs.id
    acrLoginServer: acr.outputs.loginServer
    keyVaultName: keyVault.outputs.name
    serviceName: service.name
    servicePort: service.port
    envVars: service.envVars
    useGoogleCredentials: service.?useGoogleCredentials ?? false
    minReplicas: minReplicas
    maxReplicas: maxReplicas
  }
}]

// -----------------------------------------------------------------------------
// Outputs (azd が使用)
// -----------------------------------------------------------------------------

@description('リソースグループ名')
output AZURE_RESOURCE_GROUP string = rg.name

@description('Container Registry ログインサーバー')
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = acr.outputs.loginServer

@description('Container Registry 名')
output AZURE_CONTAINER_REGISTRY_NAME string = acr.outputs.name

@description('WebUI のURL')
output SERVICE_WEBUI_URL string = 'https://${webui.outputs.fqdn}'

@description('Container Apps Environment ID')
output AZURE_CONTAINER_APPS_ENVIRONMENT_ID string = containerAppsEnv.outputs.environmentId

@description('Key Vault 名')
output AZURE_KEY_VAULT_NAME string = keyVault.outputs.name

@description('Managed Identity ID')
output AZURE_MANAGED_IDENTITY_ID string = identity.outputs.id
