// =============================================================================
// Log Analytics Workspace
// =============================================================================

@description('Log Analytics Workspace の名前')
param name string

@description('デプロイ先のリージョン')
param location string

@description('リソースに付与するタグ')
param tags object = {}

@description('ログの保持期間 (日)')
@minValue(30)
@maxValue(730)
param retentionInDays int = 30

// -----------------------------------------------------------------------------
// Resources
// -----------------------------------------------------------------------------

resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: retentionInDays
    publicNetworkAccessForIngestion: 'Disabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

// -----------------------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------------------

@description('Log Analytics Workspace のリソースID')
output workspaceId string = logAnalyticsWorkspace.id

@description('Log Analytics Workspace のカスタマーID')
output customerId string = logAnalyticsWorkspace.properties.customerId

#disable-next-line outputs-should-not-contain-secrets
@description('Log Analytics Workspace の共有キー')
output sharedKey string = logAnalyticsWorkspace.listKeys().primarySharedKey
