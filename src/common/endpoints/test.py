import json
import pickle

import requests
import numpy as np
import tensorflow as tf
from azureml.core import Webservice, Workspace
from tensorflow.keras.models import load_model


ws = Workspace.from_config()
service = Webservice(workspace=ws, name="aci-tests-height-s1")

uri = service.scoring_uri
print(requests.get(uri))

"""Load the list of depthmaps in scan as numpy array"""


def tf_load_pickle(path, max_value):
    depthmaps = []

    """Utility to load the depthmap pickle file"""
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


depthmap = tf_load_pickle(
    '/Users/prajwalsingh/cgm-ml/data/anon-depthmap-mini/scans/1583462470-16tvfmb1d0/100/pc_1583462470-16tvfmb1d0_1591122155216_100_000.p', 7.5)

headers = {"Content-Type": "application/json"}
data = {
    "data": depthmap,
}

data = json.dumps(data)

response = requests.post(uri, data=data, headers=headers)
print(response.json())
