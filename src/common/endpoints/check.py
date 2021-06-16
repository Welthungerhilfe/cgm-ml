import json
import pickle

import requests
import tensorflow as tf
from azureml.core import Webservice, Workspace

from config import CONFIG


def tf_load_pickle(path, max_value):
    depthmaps = []

    def py_load_pickle(path, max_value):
        depthmap, _ = pickle.load(open(path.numpy(), "rb"))
        depthmap = depthmap.astype("float32")
        depthmap = depthmap / max_value
        depthmap = tf.image.resize(depthmap, (240, 180))
        return depthmap, _

    depthmap, _ = tf.py_function(py_load_pickle, [path, max_value], [tf.float32, tf.float32])
    depthmap.set_shape((240, 180, 1))
    depthmap = depthmap.numpy().tolist()
    depthmaps.append(depthmap)
    return depthmaps


if __name__ == "__main__":
    if CONFIG.LOCALTEST:
        uri = 'http://localhost:6789/'
    else:
        ws = Workspace.from_config()
        service = Webservice(workspace=ws, name=CONFIG.ENDPOINT_NAME)
        uri = service.scoring_uri

    requests.get(uri)
    depthmap = tf_load_pickle(CONFIG.TEST_FILE, 7.5)

    headers = {"Content-Type": "application/json"}
    data = {
        "data": depthmap,
    }

    data = json.dumps(data)

    response = requests.post(uri, data=data, headers=headers)
    print(response.json())
