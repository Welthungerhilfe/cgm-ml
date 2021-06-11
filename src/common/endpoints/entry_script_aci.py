import json

import numpy as np
from tensorflow.keras.models import load_model


def init():
    global model
    model = load_model('/var/azureml-app/azureml-models/Deepensemble/5/outputs/best_model.ckpt/', compile=False)


def run(data):
    test = json.loads(data)
    data_np = np.array(test['data'])
    value = model.predict(data_np)
    return value.tolist()
