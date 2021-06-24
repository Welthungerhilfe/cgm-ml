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

### Setup Secrets

In the Azure Portal, put secrets into key vault:

* dset-sa-connectionstring
* dset-sa-name
* mlapi-db-host
* mlapi-db-pw
* mlapi-db-user
* mlapi-sa-connectionstring
* mlapi-sa-name

In Databricks:

* Append #secrets/createScope to databricks URL -> Create scope
	* name: 'cgm-ml-scope'
	* Manage Principal: All Users
  	* Get DNS Name and Resource ID from Azure Portal
    
#### Generating SAS tokens

To generate SAS token for access to a storage account in portal, go to the storage account in portal and from left panel select Shared access signature from Security+Networking

* Permissions for mlapi storage account
    * Read
    * List

* Permissions for dset storage account
    * Read
    * List
    * Write

for more details you can check this https://adamtheautomator.com/azure-sas-token/

### Check if setup is correct

To check if the setup is correct, you can run the databricks notebook and see if all the steps succeed.


# Databricks setup for new user

A new user should setup cgm-ml git(hub) repository

* Repos -> Add Repo
	* cgm-ml with HTTPS
	* open databricks notebook
* Create git access token
	* Go to https://github.com/settings/tokens and create a token -> remember this
* User settings -> Git integration:
	* Select: Github, Supply your github handle + token
