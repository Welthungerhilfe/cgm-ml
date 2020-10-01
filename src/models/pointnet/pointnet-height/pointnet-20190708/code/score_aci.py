import json
import numpy as np
import os
import sys

sys.path.append('/var/azureml-app/azureml-models/pointnet-height-20190708/1/pointnet')
import modelutils, utils

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType

def init():
    global model
    
    input_shape = [10000, 3]
    output_size = 1
    hidden_sizes = [512, 256, 128]
    weights_path = '/var/azureml-app/azureml-models/pointnet-height-20190708/1/pointnet/20190708-0919_2379-595height-pointnet-model-weights.h5' 
    model = modelutils.load_pointnet(weights_path, input_shape, output_size, hidden_sizes)
    
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
