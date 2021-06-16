import json
import pickle
import sys

import numpy as np
import requests
import tensorflow as tf
from azureml.core import Webservice, Workspace

from config import CONFIG

sys.path.append('./src/common/data_utilities')  # noqa
import mlpipeline_utils

if __name__ == "__main__":
    if CONFIG.LOCALTEST:
        uri = 'http://localhost:6789/'
    else:
        ws = Workspace.from_config()
        service = Webservice(workspace=ws, name=CONFIG.ENDPOINT_NAME)
        uri = service.scoring_uri

    requests.get(uri)
    depthmap = mlpipeline_utils.get_depthmaps(CONFIG.TEST_FILE).tolist()

    headers = {"Content-Type": "application/json"}
    data = {
        "data": depthmap,
    }

    data = json.dumps(data)

    response = requests.post(uri, data=data, headers=headers)
    print(response.json())
