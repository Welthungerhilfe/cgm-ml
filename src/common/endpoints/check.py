import json
import sys

import requests
from azureml.core import Webservice, Workspace

from config import CONFIG

sys.path.append('./src/common/data_utilities')  # noqa
import mlpipeline_utils  # noqa: E402

if __name__ == "__main__":
    if CONFIG.LOCALTEST:
        uri = 'http://localhost:6789/'
    else:
        workspace = Workspace.from_config()
        service = Webservice(workspace=workspace, name=CONFIG.ENDPOINT_NAME)
        uri = service.scoring_uri

    requests.get(uri)
    depthmap = mlpipeline_utils.get_depthmaps(CONFIG.TEST_FILE).tolist()  # Make JSON serializable

    headers = {"Content-Type": "application/json"}
    data = {
        "data": depthmap,
    }

    data = json.dumps(data)

    response = requests.post(uri, data=data, headers=headers)
    print(response.json())
