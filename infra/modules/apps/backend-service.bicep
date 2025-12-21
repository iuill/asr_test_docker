// =============================================================================
// Backend Service Container App (azd対応・共通テンプレート)
// =============================================================================
// 内部通信専用のバックエンドSTTサービス
// - azure-stt, azure-stt-diarization
// - google-stt-v1, google-stt-chirp2, google-stt-chirp3
// - openai-stt, openai-stt-mini
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

@description('サービス名 (イメージ名に使用)')
param serviceName string

@description('サービスのリッスンポート')
param servicePort int

@description('環境変数の配列')
param envVars array

@description('Google認証情報を使用するか')
param useGoogleCredentials bool = false

@description('最小レプリカ数')
param minReplicas int = 0

@description('最大レプリカ数')
param maxReplicas int = 3

// -----------------------------------------------------------------------------
// Variables
// -----------------------------------------------------------------------------

var keyVaultUri = 'https://${keyVaultName}${environment().suffixes.keyvaultDns}/secrets'

// 環境変数を処理 (secretRefがあればシークレット参照に変換)
var processedEnvVars = [for env in envVars: env.?secretRef != null ? {
  name: env.name
  secretRef: env.secretRef
} : {
  name: env.name
  value: env.value
}]

// シークレット定義を生成
var secretDefinitions = [for env in envVars: env.?secretRef != null ? {
  name: env.secretRef
  keyVaultUrl: '${keyVaultUri}/${env.secretRef}'
  identity: managedIdentityId
} : null]

// nullを除外
var secrets = filter(secretDefinitions, s => s != null)

// Google認証情報用シークレット
var googleSecrets = useGoogleCredentials ? [
  {
    name: 'google-credentials-base64'
    keyVaultUrl: '${keyVaultUri}/google-credentials-base64'
    identity: managedIdentityId
  }
] : []

// -----------------------------------------------------------------------------
// Resources
// -----------------------------------------------------------------------------

resource backendApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: name
  location: location
  tags: union(tags, { 'azd-service-name': serviceName })
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

      // 内部通信のみ (外部公開なし)
      ingress: {
        external: false
        targetPort: servicePort
        transport: 'auto'
      }

      // シークレット
      secrets: concat(secrets, googleSecrets)

      // ACR認証
      registries: [
        {
          server: acrLoginServer
          identity: managedIdentityId
        }
      ]
    }

    template: {
      // Google認証情報を使用する場合はinitContainerでファイル生成
      initContainers: useGoogleCredentials ? [
        {
          name: 'init-google-creds'
          image: 'mcr.microsoft.com/azure-cli:latest'
          command: [
            '/bin/sh'
            '-c'
            'echo $GOOGLE_CREDS_BASE64 | base64 -d > /secrets/google/credentials.json'
          ]
          env: [
            {
              name: 'GOOGLE_CREDS_BASE64'
              secretRef: 'google-credentials-base64'
            }
          ]
          resources: {
            cpu: json('0.25')
            memory: '0.5Gi'
          }
          volumeMounts: [
            {
              volumeName: 'google-secrets'
              mountPath: '/secrets/google'
            }
          ]
        }
      ] : []

      containers: [
        {
          name: serviceName
          image: '${acrLoginServer}/asr/${serviceName}:latest'
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: concat(processedEnvVars, [
            {
              name: 'HOST'
              value: '0.0.0.0'
            }
            {
              name: 'PORT'
              value: string(servicePort)
            }
          ])

          // Google認証情報をマウント
          volumeMounts: useGoogleCredentials ? [
            {
              volumeName: 'google-secrets'
              mountPath: '/secrets/google'
            }
          ] : []

          // ヘルスチェック
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: servicePort
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
                port: servicePort
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

      // ボリューム定義
      volumes: useGoogleCredentials ? [
        {
          name: 'google-secrets'
          storageType: 'EmptyDir'
        }
      ] : []

      // スケーリング設定
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '50'
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

@description('Container App の内部FQDN')
output fqdn string = backendApp.properties.configuration.ingress.fqdn

@description('Container App のリソースID')
output id string = backendApp.id

@description('Container App の名前')
output name string = backendApp.name
