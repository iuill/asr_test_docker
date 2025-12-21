// =============================================================================
// Azure Container Registry
// =============================================================================

@description('Container Registry の名前 (グローバルで一意)')
param name string

@description('デプロイ先のリージョン')
param location string

@description('リソースに付与するタグ')
param tags object = {}

@description('ACRにアクセスするManaged IdentityのプリンシパルID')
param managedIdentityPrincipalId string

@description('SKU (Basic, Standard, Premium)')
@allowed(['Basic', 'Standard', 'Premium'])
param sku string = 'Basic'

// -----------------------------------------------------------------------------
// Resources
// -----------------------------------------------------------------------------

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: name
  location: location
  tags: tags
  sku: {
    name: sku
  }
  properties: {
    adminUserEnabled: false
    publicNetworkAccess: 'Enabled'
  }
}

// AcrPull ロールを Managed Identity に付与
resource acrPullRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, managedIdentityPrincipalId, 'acrpull')
  scope: acr
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d') // AcrPull
    principalId: managedIdentityPrincipalId
    principalType: 'ServicePrincipal'
  }
}

// -----------------------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------------------

@description('Container Registry のログインサーバー')
output loginServer string = acr.properties.loginServer

@description('Container Registry のリソースID')
output id string = acr.id

@description('Container Registry の名前')
output name string = acr.name
