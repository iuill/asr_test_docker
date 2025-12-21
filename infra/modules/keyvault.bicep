// =============================================================================
// Azure Key Vault
// =============================================================================

@description('Key Vault の名前')
param name string

@description('デプロイ先のリージョン')
param location string

@description('リソースに付与するタグ')
param tags object = {}

@description('Key Vaultにアクセスする Managed Identity のプリンシパルID')
param managedIdentityPrincipalId string

@description('保存するシークレットの配列')
param secrets array = []

// -----------------------------------------------------------------------------
// Resources
// -----------------------------------------------------------------------------

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 90
    publicNetworkAccess: 'Disabled'
  }
}

// Key Vault Secrets User ロールを Managed Identity に付与
resource keyVaultSecretsUserRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, managedIdentityPrincipalId, 'secretsuser')
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')
    principalId: managedIdentityPrincipalId
    principalType: 'ServicePrincipal'
  }
}

// シークレットを作成
resource keyVaultSecrets 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = [for secret in secrets: {
  parent: keyVault
  name: secret.name
  properties: {
    value: secret.value
  }
}]

// -----------------------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------------------

@description('Key Vault のリソースID')
output id string = keyVault.id

@description('Key Vault の名前')
output name string = keyVault.name

@description('Key Vault のURI')
output vaultUri string = keyVault.properties.vaultUri
