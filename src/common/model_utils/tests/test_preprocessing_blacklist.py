from pathlib import Path
import pytest
import sys

sys.path.append(str(Path(__file__).parents[2]))  # common/ dir

from model_utils.preprocessing import filter_blacklisted_qrcodes
from model_utils.tests.test_preprocessing_constants import BLACKLIST_QRCODES_8SAMPLES, BLACKLIST_QRCODES_WRONG_SIZE, BLACKLIST_QRCODES_EMPTY


def test_always_passes():
    assert True


"""def test_always_fails():
    assert False"""


"""def filter_blacklisted_qrcodes(qrcode_paths):
    qrcode_paths_filtered = []
    for qrcode_path in qrcode_paths:
        qrcode_str = qrcode_path.split('/')[-1]
        assert '-' in qrcode_str and len(qrcode_str) == 21, qrcode_str
        if qrcode_str in BLACKLIST_QRCODES:
            continue
        qrcode_paths_filtered.append(qrcode_path)
    return qrcode_paths_filtered"""

"""so wirds aufgerufen
    qrcode_paths = filter_blacklisted_qrcodes(qrcode_paths)"""