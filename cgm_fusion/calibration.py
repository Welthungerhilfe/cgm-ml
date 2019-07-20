import numpy
import xmltodict
import numpy as np

import os
calib_file = r'/whhdata/calibration.xml' # todo make this a parameter

def get_intrinsic_matrix():
    with open(calib_file) as fd:
        calib = xmltodict.parse(fd.read())

    arr = calib['rig']['camera'][1]['camera_model']['params']
    fu, fv, u0, v0, k1, k2, k3 = np.fromstring(arr.replace('[', '').replace(']', ''), sep=';')
    _gamma = 1
    intrinsic = np.array(
        [[fu, _gamma, u0, 0],
         [0, fv, v0, 0],
         [0, 0, 1, 0]])
    return intrinsic


def get_k():
    with open(calib_file) as fd:
        calib = xmltodict.parse(fd.read())

    arr = calib['rig']['camera'][1]['camera_model']['params']
    fu, fv, u0, v0, k1, k2, k3 = np.fromstring(arr.replace('[', '').replace(']', ''), sep=';')
    _gamma = 1
    intrinsic = np.array(
        [[fu, 0, u0, 0],
         [0, fv, v0, 0],
         [0, 0, 1, 0]])
    return k1, k2, k3


def get_extrinsic_matrix(idx=1):
    with open(calib_file) as fd:
        calib = xmltodict.parse(fd.read())
        
    arr = calib['rig']['extrinsic_calibration'][idx]['A_T_B']
    arr = arr.split(';')
    arr = [x.replace('[', '').replace(']', '') for x in arr]
    mat = np.array([np.fromstring(x, sep=',') for x in arr])
    mat[:3, :3] = mat[:3, :3].T # maybe transpose?
    mat = np.vstack([mat, np.array([0, 0, 0, 1])])
    return mat
    