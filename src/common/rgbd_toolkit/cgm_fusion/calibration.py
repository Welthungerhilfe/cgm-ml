# Child Growth Monitor - Free Software for Zero Hunger
# Copyright (c) 2019 Dr. Christian Pfitzner <christian.pfitzner@th-nuernberg.de> for Welthungerhilfe
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np
import xmltodict

#calib_file = r'/whhdata/calibration.xml' # todo make this a parameter


def get_intrinsic_matrix(calib_file):
    with open(calib_file) as fd:
        calib = xmltodict.parse(fd.read())

    arr = calib['rig']['camera'][1]['camera_model']['params']
    fu, fv, u0, v0, k1, k2, k3 = np.fromstring(arr.replace('[', '').replace(
        ']', ''),
        sep=';')
    _gamma = 1
    intrinsic = np.array([[fu, _gamma, u0, 0], [0, fv, v0, 0], [0, 0, 1, 0]])
    return intrinsic


def get_intrinsic_matrix_depth(calib_file):
    with open(calib_file) as fd:
        calib = xmltodict.parse(fd.read())

    arr = calib['rig']['camera'][0]['camera_model']['params']
    fu, fv, u0, v0, k1, k2, p1, p2, k3 = np.fromstring(arr.replace(
        '[', '').replace(']', ''),
        sep=';')
    _gamma = 1
    intrinsic = np.array([[fu, _gamma, u0, 0], [0, fv, v0, 0], [0, 0, 1, 0]])
    return intrinsic


def get_k(calib_file):
    with open(calib_file) as fd:
        calib = xmltodict.parse(fd.read())

    arr = calib['rig']['camera'][1]['camera_model']['params']
    fu, fv, u0, v0, k1, k2, k3 = np.fromstring(arr.replace('[', '').replace(
        ']', ''),
        sep=';')
    np.array([[fu, 0, u0, 0], [0, fv, v0, 0], [0, 0, 1, 0]])
    return k1, k2, k3


def get_k_depth(calib_file):
    with open(calib_file) as fd:
        calib = xmltodict.parse(fd.read())

    arr = calib['rig']['camera'][0]['camera_model']['params']
    fu, fv, u0, v0, k1, k2, p1, p2, k3 = np.fromstring(arr.replace(
        '[', '').replace(']', ''),
        sep=';')
    np.array([[fu, 0, u0, 0], [0, fv, v0, 0], [0, 0, 1, 0]])
    return k1, k2, k3


def get_extrinsic_matrix(calib_file, idx=1):
    with open(calib_file) as fd:
        calib = xmltodict.parse(fd.read())

    arr = calib['rig']['extrinsic_calibration'][idx]['A_T_B']
    arr = arr.split(';')
    arr = [x.replace('[', '').replace(']', '') for x in arr]
    mat = np.array([np.fromstring(x, sep=',') for x in arr])
    mat[:3, :3] = mat[:3, :3].T  # maybe transpose?
    mat = np.vstack([mat, np.array([0, 0, 0, 1])])
    return mat


def get_extrinsic_matrix_depth(calib_file, idx=1):
    with open(calib_file) as fd:
        calib = xmltodict.parse(fd.read())

    arr = calib['rig']['extrinsic_calibration'][0]['A_T_B']
    arr = arr.split(';')
    arr = [x.replace('[', '').replace(']', '') for x in arr]
    mat = np.array([np.fromstring(x, sep=',') for x in arr])
    mat[:3, :3] = mat[:3, :3].T  # maybe transpose?
    mat = np.vstack([mat, np.array([0, 0, 0, 1])])
    return mat
