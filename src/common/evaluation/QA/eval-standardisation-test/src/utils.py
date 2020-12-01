import math
import os
import pickle
from pathlib import Path
import sys
import zipfile

from azureml.core import Experiment, Run
import glob2 as glob
import numpy as np
import pandas as pd
from skimage.transform import resize

sys.path.append(str(Path(__file__).parents[0]))

from qa_config import DATA_CONFIG, EVAL_CONFIG, RESULT_CONFIG  # noqa: E402

image_target_height = 240
image_target_width = 180

measure_name_list = ['HEIGHT', 'WEIGHT', 'MUAC']
measure_category_list = ['GOOD', 'ACCEPTABLE', 'POOR', 'REJECT']
permissible_measure = {'HEIGHT':
                       {'GOOD': 0.4, 'ACCEPTABLE': 0.6, 'POOR': 1.0, 'REJECT': None},
                       'WEIGHT':
                       {'GOOD': 0.04, 'ACCEPTABLE': 0.10, 'POOR': 0.21, 'REJECT': None},
                       'MUAC':
                       {'GOOD': 2.0, 'ACCEPTABLE': 2.7, 'POOR': 3.3, 'REJECT': None}
                       }


def get_intra_TEM(measure_one, measure_two):
    '''
    https://www.scielo.br/pdf/rbme/v11n1/en_24109.pdf
    Compute Intra Technical Error of Measurement
    '''
    assert(len(measure_one.index) == len(measure_two.index))
    sum_of_square_of_deviation = ((measure_one - measure_two) ** 2).sum()
    absolute_TEM = math.sqrt(sum_of_square_of_deviation / (2 * len(measure_one.index)))

    if EVAL_CONFIG.DEBUG_LOG:
        print("Absolute TEM : ", absolute_TEM)

    return absolute_TEM


def get_measure_category(technical_error_of_measurement, measure_name):
    '''
    Return the measure category based on the technical error of measurement
    e.g. GOOD, ACCEPTABLE, POOR, REJECT
    '''
    if technical_error_of_measurement < permissible_measure[measure_name]['GOOD']:
        measure_category = 'GOOD'
    elif technical_error_of_measurement < permissible_measure[measure_name]['ACCEPTABLE']:
        measure_category = 'ACCEPTABLE'
    elif technical_error_of_measurement < permissible_measure[measure_name]['POOR']:
        measure_category = 'POOR'
    else:
        measure_category = 'REJECT'

    return measure_category


def load_depth(filename):
    with zipfile.ZipFile(filename) as z:
        with z.open('data') as f:
            line = str(f.readline())[2:-3]
            header = line.split("_")
            res = header[0].split("x")
            #print(res)
            width = int(res[0])
            height = int(res[1])
            depthScale = float(header[1])
            maxConfidence = float(header[2])
            data = f.read()
            f.close()
        z.close()
    return data, width, height, depthScale, maxConfidence


def parse_depth(tx, ty, data, depthScale):
    depth = data[(int(ty) * width + int(tx)) * 3 + 0] << 8
    depth += data[(int(ty) * width + int(tx)) * 3 + 1]
    depth *= depthScale
    return depth


def prepare_depthmap(data, width, height, depthScale):
    # prepare array for output
    output = np.zeros((width, height, 1))
    for cx in range(width):
        for cy in range(height):
            #             output[cx][height - cy - 1][0] = parse_confidence(cx, cy)
            #             output[cx][height - cy - 1][1] = im_array[cy][cx][1] / 255.0 #test matching on RGB data
            # output[cx][height - cy - 1][2] = 1.0 - min(parse_depth(cx, cy) / 2.0,
            # 1.0) #depth data scaled to be visible
            output[cx][height - cy - 1] = parse_depth(cx, cy, data, depthScale)  # depth data scaled to be visible
    return (np.array(output, dtype='float32').reshape(width, height), height, width)


def preprocess(depthmap):
    #print(depthmap.dtype)
    depthmap = preprocess_depthmap(depthmap)
    #depthmap = depthmap/depthmap.max()
    depthmap = depthmap / 7.5
    depthmap = resize(depthmap, (image_target_height, image_target_width))
    depthmap = depthmap.reshape((depthmap.shape[0], depthmap.shape[1], 1))
    #depthmap = depthmap[None, :]
    return depthmap


def setWidth(value):
    global width
    width = value

def setHeight(value):
    global height
    height = value
