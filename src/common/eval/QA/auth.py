import argparse
from azureml.core import Workspace
from azureml.core.authentication import AuthenticationException, AzureCliAuthentication


def get_auth():
    '''
    Authentication to access workspace
    '''
    try:
        auth = AzureCliAuthentication()
        auth.get_authentication_header()
    except AuthenticationException:
        print("Authentication Error Occured")

    return auth


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sid", "--subscription_id", help="Subscription ID")
    parser.add_argument("-rg", "--resource_group", help="Resource Group")
    parser.add_argument("-wn", "--workspace_name", help="Workspace  Name")

    args = parser.parse_args()

    ws = Workspace(subscription_id=args.subscription_id,
                   resource_group=args.resource_group,
                   workspace_name=args.workspace_name,
                   auth=get_auth())

    print("Workspace Details")
    print(ws.get_details())

    print("Success of Authentication and Workspace Setup")

    ws.write_config()
    print("Saved config file")
