from pathlib import Path
import sys
import os
import pytest

CWD = Path(__file__).resolve()
sys.path.append(str(CWD.parents[2]))  # common/ dir

from model_utils.preprocessing import filter_blacklisted_qrcodes  # noqa: E402


BASE_PATH = "this/is/the/test/path"


def prepare_qrs():
    qrcodes_samples = [
        "1585000019-syglokl9nx",
        "1585366118-qao4zsk0m3",
        "1585360775-fa64muouel",
        '1583855791-ldfc59ywg5',
        '1583997882-3jqstr1119',
        '1584998372-d85ogmqucw',
        '1585274424-3oqa4i262a',
        '1585010027-xb21f31tvj',
        '0000000000-0000000000',
        '1111111111-1111111111',
        '2222222222-2222222222',
    ]
    qrcode_paths = []
    for qrcode in qrcodes_samples:
        qrcode_path = os.path.join(BASE_PATH + '/' + qrcode)
        qrcode_paths.append(qrcode_path)
    return qrcode_paths


def test_filter_blacklisted_qrcodes():
    qrcode_paths = prepare_qrs()
    filtered_qrcode_paths = filter_blacklisted_qrcodes(qrcode_paths)
    assert (len(filtered_qrcode_paths) == 3)


def prepare_wrong_qrs():
    qrcodes_wrong_size = [
        "1585000019-syglokl9nx",
        "1585366118-qao4zsk0m3",
        "0000000000-0000000000",
        "1111111111-1111111111",
        "1585360775-fa64muouelXXXX",
        "2222222222-2222222222",
    ]
    qrcode_paths_wrong = []
    for qrcode in qrcodes_wrong_size:
        qrcode_path = os.path.join(BASE_PATH + '/' + qrcode)
        qrcode_paths_wrong.append(qrcode_path)
    print("qrcodes_paths_wrong", qrcode_paths_wrong)
    return qrcode_paths_wrong


def test_filter_qrcodes_wrong_size():
    qrcode_paths_wrong = prepare_wrong_qrs()
    with pytest.raises(AssertionError) as e:
        filter_blacklisted_qrcodes(qrcode_paths_wrong)
    assert str(e.value) == "1585360775-fa64muouelXXXX", "first wrong qrcode"


def prepare_empty_qrs():
    qrcodes_empty = []
    qrcode_paths_empty = []
    for qrcode in qrcodes_empty:
        qrcode_path = os.path.join(BASE_PATH + '/' + qrcode)
        qrcode_paths_empty.append(qrcode_path)
    return qrcode_paths_empty


def test_filter_qrcodes_empty():
    qrcode_paths_empty = prepare_empty_qrs()
    with pytest.raises(Exception) as e:
        filter_blacklisted_qrcodes(qrcode_paths_empty)
    assert str(e.value) == "The provided qrcode_path is empty"
