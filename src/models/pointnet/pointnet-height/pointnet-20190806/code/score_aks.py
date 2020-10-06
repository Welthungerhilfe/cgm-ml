import utils
import modelutils
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.schema_decorators import input_schema, output_schema
import tensorflow as tf
import json
import numpy as np
import os
import sys

sys.path.append('/structure/azureml-app/azureml-models/pointnet-height-20190806/1/pointnet')
#sys.path.append('/var/azureml-app/azureml-models/20190806-1551_220-55height/1/20190806-1551_220-55height')

tf.compat.v1.disable_eager_execution()


def init():
    global model
    input_shape = [10000, 3]
    output_size = 1
    hidden_sizes = [512, 256, 128]
    weights_path = '/structure/azureml-app/azureml-models/pointnet-height-20190806/1/pointnet/20190806-1551_220-55height-pointnet-model-weights.h5'
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
