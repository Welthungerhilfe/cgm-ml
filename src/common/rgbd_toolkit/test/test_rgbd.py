import sys
from pathlib import Path

import numpy as np
sys.path.append(str(Path(__file__).parents[1])) # noqa
from rgbd import find_closest, get_files # noqa


REPO_DIR = str(Path(__file__).parents[0].absolute())


def test_find_closest_100():
    rgb = np.asarray([668549.91427676, 668549.98114676, 668578.70910376, 668601.94653876,
                      668602.31434576])
    pcd = 668550.01378682
    idx = find_closest(rgb, pcd)
    assert idx == 1


def test_find_closest_101():
    rgb = np.asarray([668549.91427676, 668549.98114676, 668578.70910376, 668601.94653876,
                      668602.31434576])
    pcd = 668601.96653282
    idx = find_closest(rgb, pcd)
    assert idx == 3


def test_find_closest_102():
    rgb = np.asarray([668549.91427676, 668549.98114676, 668578.70910376, 668601.94653876,
                      668602.31434576])
    pcd = 668578.83466482
    idx = find_closest(rgb, pcd)
    assert idx == 2


def test_get_files():
    norm_rgb_time = np.asarray([668549.91427676, 668601.94653876, 668578.70910376, 668549.98114676, 668602.31434576])
    rgb_path = [REPO_DIR + '/qr_code/qr_code_test/rgb/rgb_1584994919-58mqazoorz_1591550095148_100_668549.914276759.jpg', REPO_DIR + '/qr_code/qr_code_test/rgb/rgb_1584994919-58mqazoorz_1591550095148_101_668601.946538759.jpg',
                REPO_DIR + '/qr_code/qr_code_test/rgb/rgb_1584994919-58mqazoorz_1591550095148_102_668578.709103759.jpg', REPO_DIR + '/qr_code/qr_code_test/rgb/rgb_1584994919-58mqazoorz_1591550095148_100_668549.981146759.jpg', REPO_DIR + '/qr_code/qr_code_test/rgb/rgb_1584994919-58mqazoorz_1591550095148_101_668602.314345759.jpg']
    norm_pcd_time = np.asarray([668578.83466482, 668550.01378682, 668601.96653282])
    pcd_path = [REPO_DIR + '/qr_code/qr_code_test/pc/pc_1584994919-58mqazoorz_1591550095148_102_000.pcd',
                REPO_DIR + '/qr_code/qr_code_test/pc/pc_1584994919-58mqazoorz_1591550095148_100_000.pcd', REPO_DIR + '/qr_code/qr_code_test/pc/pc_1584994919-58mqazoorz_1591550095148_101_000.pcd']
    files = get_files(norm_rgb_time, rgb_path, norm_pcd_time, pcd_path)

    assert files == [[REPO_DIR + '/qr_code/qr_code_test/pc/pc_1584994919-58mqazoorz_1591550095148_102_000.pcd', REPO_DIR + '/qr_code/qr_code_test/rgb/rgb_1584994919-58mqazoorz_1591550095148_102_668578.709103759.jpg'], [REPO_DIR + '/qr_code/qr_code_test/pc/pc_1584994919-58mqazoorz_1591550095148_100_000.pcd',
                                                                                                                                                                                                                           REPO_DIR + '/qr_code/qr_code_test/rgb/rgb_1584994919-58mqazoorz_1591550095148_100_668549.981146759.jpg'], [REPO_DIR + '/qr_code/qr_code_test/pc/pc_1584994919-58mqazoorz_1591550095148_101_000.pcd', REPO_DIR + '/qr_code/qr_code_test/rgb/rgb_1584994919-58mqazoorz_1591550095148_101_668601.946538759.jpg']]
