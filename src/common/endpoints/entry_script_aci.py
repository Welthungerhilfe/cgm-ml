import json

import numpy as np
from tensorflow.keras.models import load_model

from config import CONFIG


def init():
    global model
    model = load_model(
        str('/var/azureml-app/azureml-models' / CONFIG.MODEL_NAME / CONFIG.VERSION / 'outputs/best_model.ckpt/'), compile=False)


def run(data):
    test = json.loads(data)
    data_np = np.array(test['data'])
    value = model.predict(data_np)
    return value.tolist()
