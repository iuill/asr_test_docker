// =============================================================================
// WebUI Container App (azd対応)
// =============================================================================
// 外部公開されるフロントエンド兼プロキシサービス
// - HTTPS (443) で外部公開
// - JWT認証機能
// - WebSocket対応
// - バックエンドサービスへのプロキシ
// =============================================================================

@description('Container App の名前')
param name string

@description('デプロイ先のリージョン')
param location string

@description('リソースに付与するタグ')
param tags object = {}

@description('Container Apps Environment のID')
param environmentId string

@description('User-Assigned Managed Identity のリソースID')
param managedIdentityId string

@description('ACR のログインサーバー')
param acrLoginServer string

@description('Key Vault の名前')
param keyVaultName string

@description('認証を有効にするか')
param authEnabled bool = true

@description('最小レプリカ数')
param minReplicas int = 0

@description('最大レプリカ数')
param maxReplicas int = 3

@description('カスタムドメイン')
param customDomain string = ''

@description('バックエンドサービスの定義')
param backendServices array

// -----------------------------------------------------------------------------
// Variables
// -----------------------------------------------------------------------------

var keyVaultUri = 'https://${keyVaultName}${environment().suffixes.keyvaultDns}/secrets'

// バックエンドサービスのURL環境変数を生成
// config.py が期待する形式: AZURE_STT_URL, OPENAI_STT_URL など
var backendUrlEnvVars = [for service in backendServices: {
  name: '${replace(toUpper(service.name), '-', '_')}_URL'
  value: 'http://${replace(name, 'webui', service.shortName)}' // 内部DNS名 (shortNameを使用)
}]

// -----------------------------------------------------------------------------
// Resources
// -----------------------------------------------------------------------------

resource webuiApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: name
  location: location
  tags: union(tags, { 'azd-service-name': 'webui' })
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentityId}': {}
    }
  }
  properties: {
    managedEnvironmentId: environmentId
    workloadProfileName: 'Consumption'

    configuration: {
      activeRevisionsMode: 'Single'

      // 外部公開設定
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false  // HTTPSのみ
        // カスタムドメイン設定 (オプション)
        customDomains: customDomain != '' ? [
          {
            name: customDomain
            bindingType: 'SniEnabled'
          }
        ] : []
      }

      // シークレット (Key Vault参照)
      secrets: [
        {
          name: 'jwt-secret-key'
          keyVaultUrl: '${keyVaultUri}/jwt-secret-key'
          identity: managedIdentityId
        }
        {
          name: 'auth-username'
          keyVaultUrl: '${keyVaultUri}/auth-username'
          identity: managedIdentityId
        }
        {
          name: 'auth-password'
          keyVaultUrl: '${keyVaultUri}/auth-password'
          identity: managedIdentityId
        }
      ]

      // ACR認証
      registries: [
        {
          server: acrLoginServer
          identity: managedIdentityId
        }
      ]
    }

    template: {
      containers: [
        {
          name: 'webui'
          image: '${acrLoginServer}/asr/webui:latest'
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: concat([
            // 認証設定
            {
              name: 'AUTH_ENABLED'
              value: string(authEnabled)
            }
            {
              name: 'JWT_SECRET_KEY'
              secretRef: 'jwt-secret-key'
            }
            {
              name: 'AUTH_USERNAME'
              secretRef: 'auth-username'
            }
            {
              name: 'AUTH_PASSWORD'
              secretRef: 'auth-password'
            }
            // サーバー設定
            {
              name: 'HOST'
              value: '0.0.0.0'
            }
            {
              name: 'PORT'
              value: '8000'
            }
          ], backendUrlEnvVars)

          // ヘルスチェック
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: 8000
                scheme: 'HTTP'
              }
              initialDelaySeconds: 10
              periodSeconds: 30
              timeoutSeconds: 5
              failureThreshold: 3
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/health'
                port: 8000
                scheme: 'HTTP'
              }
              initialDelaySeconds: 5
              periodSeconds: 10
              timeoutSeconds: 3
              failureThreshold: 3
            }
          ]
        }
      ]

      // スケーリング設定
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '100'
              }
            }
          }
        ]
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------------------

@description('Container App のFQDN')
output fqdn string = webuiApp.properties.configuration.ingress.fqdn

@description('Container App のリソースID')
output id string = webuiApp.id

@description('Container App の名前')
output name string = webuiApp.name

@description('Container App の最新リビジョン名')
output latestRevisionName string = webuiApp.properties.latestRevisionName
