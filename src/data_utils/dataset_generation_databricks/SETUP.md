# Setup Databricks in a new environment

## Create resources

* Create a Key Vault resource
* Create Storage Account (SA) resource and create one container called "cgm-datasets"
* Create Databricks resource

## Setup/Configure Databricks

In the Azure Portal, go to the Databricks resource and find Databricks URL.
Visit that URL and login to Databricks.

### Setup git(hub) and repository

* Admin Console -> Enable Repos
* Repos -> Add Repo
	* cgm-ml with HTTPS
	* open databricks notebook
* Create git access token
	* Go to https://github.com/settings/tokens and create a token -> remember this
* User settings -> Git integration:
	* Select: Github, Supply your github handle + token

### Setup Compute

* Compute
    * Create Cluster 'cgm-ml-cluster': Runtime 7.6 (CPU)
	* Libraries: Install the following libraries from PyPI
		* scikit-image
		* cgm-ml-common
        
---
**NOTE**

If you get error following error
```
Code: MissingSubscriptionRegistration
Message: The subscription is not registered to use namespace {resource-provider-namespace}
```

follow https://docs.microsoft.com/en-us/azure/azure-resource-manager/templates/error-register-resource-provider#solution-3---azure-portal

---

### Setup Compute

In the Azure Portal, put secrets into key vault:

* dset-sa-sas-token
* mlapi-db-host
* mlapi-db-pw
* mlapi-db-user
* mlapi-sa-sas-token

In Databricks:

* Append #secrets/createScope to databricks URL -> Create scope
	* name: 'cgm-ml-scope'
	* Manage Principal: All Users
  	* Get DNS Name and Resource ID from Azure Portal

### Check if setup is correct

To check if the setup is correct, you can run the databricks notebook and see if all the steps succeed.
