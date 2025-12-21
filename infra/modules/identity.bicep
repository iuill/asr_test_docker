// =============================================================================
// User-Assigned Managed Identity
// =============================================================================

@description('Managed Identity の名前')
param name string

@description('デプロイ先のリージョン')
param location string

@description('リソースに付与するタグ')
param tags object = {}

// -----------------------------------------------------------------------------
// Resources
// -----------------------------------------------------------------------------

resource managedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: name
  location: location
  tags: tags
}

// -----------------------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------------------

@description('Managed Identity のリソースID')
output id string = managedIdentity.id

@description('Managed Identity のプリンシパルID')
output principalId string = managedIdentity.properties.principalId

@description('Managed Identity のクライアントID')
output clientId string = managedIdentity.properties.clientId
