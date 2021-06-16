import json

import numpy as np
from tensorflow.keras.models import load_model

MODEL = None


def init():
    global MODEL
    MODEL = load_model(
        '/var/azureml-app/azureml-models/2021q1-depthmap-ensemble-height-95k/1/outputs/best_model.ckpt/', compile=False)


def run(input: json) -> list:
    data_json = json.loads(input)
    data_np = np.array(data_json['data'])
    result = MODEL.predict(data_np)
    return result.tolist()
