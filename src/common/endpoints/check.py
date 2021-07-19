import json
import sys
from pathlib import Path

import requests
from azureml.core import Webservice, Workspace

from config_deepensemble_1 import CONFIG

sys.path.append(str(Path(__file__).parents[1] / 'data_utilities'))  # noqa
import mlpipeline_utils  # noqa: E402

if __name__ == "__main__":
    if CONFIG.LOCALTEST:
        uri = 'http://localhost:6789/'
    else:
        workspace = Workspace.from_config()
        service = Webservice(workspace=workspace, name=CONFIG.ENDPOINT_NAME)
        uri = service.scoring_uri

    requests.get(uri)
    depthmap = mlpipeline_utils.get_depthmaps(CONFIG.TEST_FILES).tolist()  # Make JSON serializable

    headers = {"Content-Type": "application/json"}
    data = {
        "data": depthmap,
    }

    data = json.dumps(data)

    response = requests.post(uri, data=data, headers=headers)
    print(response.json())
