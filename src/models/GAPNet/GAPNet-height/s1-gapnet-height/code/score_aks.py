from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.schema_decorators import input_schema, output_schema
import tensorflow as tf
from GAPNet.models import GAPNet
import json
import numpy as np
import os
import sys

sys.path.append('/structure/azureml-app/azureml-models/gapnet_height_s1/1/')

#from tensorflow.keras import models


def init():
    global model
    model = GAPNet()
    output_directory = '/structure/azureml-app/azureml-models/gapnet_height_s1/1/GAPNet'
    model.load_weights(os.path.join(output_directory, "gapnet_weights.h5"))


def run(data):
    try:
        data_list = json.loads(data)
        data_np = np.array(data_list['data'])
        result = model.predict(data_np)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
