// =============================================================================
// Container Apps Environment
// =============================================================================

@description('Container Apps Environment の名前')
param name string

@description('デプロイ先のリージョン')
param location string

@description('リソースに付与するタグ')
param tags object = {}

@description('Log Analytics Workspace のリソースID')
param logAnalyticsWorkspaceId string

@secure()
@description('Log Analytics Workspace の共有キー')
param logAnalyticsSharedKey string

// -----------------------------------------------------------------------------
// Resources
// -----------------------------------------------------------------------------

resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: reference(logAnalyticsWorkspaceId, '2022-10-01').customerId
        sharedKey: logAnalyticsSharedKey
      }
    }
    zoneRedundant: false  // コスト削減のため無効 (本番では有効化推奨)
    workloadProfiles: [
      {
        name: 'Consumption'
        workloadProfileType: 'Consumption'
      }
    ]
  }
}

// -----------------------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------------------

@description('Container Apps Environment のリソースID')
output environmentId string = containerAppsEnvironment.id

@description('Container Apps Environment の名前')
output name string = containerAppsEnvironment.name

@description('Container Apps Environment のデフォルトドメイン')
output defaultDomain string = containerAppsEnvironment.properties.defaultDomain

@description('Container Apps Environment の静的IP')
output staticIp string = containerAppsEnvironment.properties.staticIp
