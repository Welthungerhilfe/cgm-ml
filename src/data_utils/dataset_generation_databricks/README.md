# Dataset Creation using DataBricks

## Security concept

Databricks may access key vault (see KV access policy: list, get)
([setup docs](https://docs.microsoft.com/en-us/azure/databricks/scenarios/store-secrets-azure-key-vault))
In the Databricks code, we get secrets (via secret scopes) that access keys in key vault.
Additionally we use SAS token for SA/Container access (not SA/Container access): there get renewed every 3month

Databricks reads from:
* only read+list
* for cgm-result container directly
