import argparse
import logging
import logging.config
from azureml.core import Workspace
from azureml.core.authentication import AuthenticationException, AzureCliAuthentication

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_auth():
    '''
    Azure CLI Authentication to access workspace
    '''
    try:
        auth = AzureCliAuthentication()
        auth.get_authentication_header()
    except AuthenticationException:
        logger.info("Authentication Error Occured")

    return auth


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sid", "--subscription_id", help="Subscription ID")
    parser.add_argument("-rg", "--resource_group", help="Resource Group")
    parser.add_argument("-wn", "--workspace_name", help="Workspace  Name")

    args = parser.parse_args()

    workspace = Workspace(subscription_id=args.subscription_id,
                          resource_group=args.resource_group,
                          workspace_name=args.workspace_name,
                          auth=get_auth())

    logger.info("Workspace Details")
    logger.info(workspace.get_details())

    logger.info("Success of Authentication and Workspace Setup")

    workspace.write_config()
    logger.info("Saved config file")
