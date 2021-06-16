import argparse
import logging
import logging.config
from azureml.core import Workspace
from azureml.core.authentication import AuthenticationException, AzureCliAuthentication

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')


def get_auth():
    """Authentication to access workspace"""
    try:
        auth = AzureCliAuthentication()
        auth.get_authentication_header()
    except AuthenticationException:
        logging.info("Authentication Error Occured")

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

    logging.info("Workspace Details")
    logging.info(workspace.get_details())

    logging.info("Success of Authentication and Workspace Setup")

    workspace.write_config()
    logging.info("Saved config file")
