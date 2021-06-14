from pathlib import Path
import sys
import os
import pytest

here = Path(__file__).resolve()
sys.path.append(str(here.parents[2]))  # common/ dir

from model_utils.preprocessing import filter_blacklisted_qrcodes  # noqa: E402

#sys.path.append(str(Path(__file__).parents[2]))  # common/ dir THIS DOES NOT WORK. GIVES ME INDEXERROR: 2


base_path = "this/is/the/test/path"


# 8 qrs below are in blacklist, 3 qrs are not in blacklist
QRCODES_11SAMPLES = [
    "1585000019-syglokl9nx",  # only to test (part of mini)
    "1585366118-qao4zsk0m3",  # in anon-depthmap-95k, child_height = 12.7, scans/1585366118-qao4zsk0m3/102/pc_1585366118-qao4zsk0m3_1593021766372_102_026.p'  # noqa: E501
    "1585360775-fa64muouel",  # in anon-depthmap-95k, child_height = 7.9, scans/1585360775-fa64muouel/202/pc_1585360775-fa64muouel_1597205960827_202_002.p',  # noqa: E501
    '1583855791-ldfc59ywg5',  # in anon-depthmap-95k, child_height
    '1583997882-3jqstr1119',  # in anon-depthmap-95k, child_height
    '1584998372-d85ogmqucw',  # in anon-depthmap-95k, child_height
    '1585274424-3oqa4i262a',  # in anon-depthmap-95k, child_height
    '1585010027-xb21f31tvj',  # in anon-depthmap-95k, pixel_value_max = 714286.0,
    '0000000000-0000000000',
    '1111111111-1111111111',
    '2222222222-2222222222',
]


qrcode_paths = []
for qrcode in QRCODES_11SAMPLES:
    qrcode_path = os.path.join(base_path + '/' + qrcode)
    qrcode_paths.append(qrcode_path)


def test_filter_blacklisted_qrcodes():
    filtered_qrcode_paths = filter_blacklisted_qrcodes(qrcode_paths)
    assert (len(filtered_qrcode_paths) == 3)


# 2 qrs in blacklist, 4 not out of which 1 is too long
QRCODES_WRONG_SIZE = [
    "1585000019-syglokl9nx",  # only to test (part of mini)
    "1585366118-qao4zsk0m3",  # in anon-depthmap-95k, child_height = 12.7, scans/1585366118-qao4zsk0m3/102/pc_1585366118-qao4zsk0m3_1593021766372_102_026.p'  # noqa: E501
    "0000000000-0000000000",
    "1111111111-1111111111",
    "1585360775-fa64muouelXXXX",  # in anon-depthmap-95k, child_height = 7.9, scans/1585360775-fa64muouel/202/pc_1585360775-fa64muouel_1597205960827_202_002.p',  # noqa: E501
    "2222222222-2222222222",
]


qrcode_paths_wrong = []
for qrcode in QRCODES_WRONG_SIZE:
    qrcode_path = os.path.join(base_path + '/' + qrcode)
    qrcode_paths_wrong.append(qrcode_path)


# maybe test for too short too
def test_filter_qrcodes_wrong_size():
    with pytest.raises(AssertionError) as e:
        filter_blacklisted_qrcodes(qrcode_paths_wrong)
    assert str(e.value) == "1585360775-fa64muouelXXXX"  # first wrong qrcode


QRCODES_EMPTY = []
qrcode_paths_empty = []
for qrcode in QRCODES_EMPTY:
    qrcode_path = os.path.join(base_path + '/' + qrcode)
    qrcode_paths_empty.append(qrcode_path)


def test_filter_qrcodes_empty():
    filtered_qrcode_paths = filter_blacklisted_qrcodes(qrcode_paths_empty)
    assert (len(filtered_qrcode_paths) == 0)


def test_always_passes():
    assert True


"""def test_always_fails():
    assert False"""
