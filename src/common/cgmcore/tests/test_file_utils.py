import shutil
from pathlib import Path
import pytest
import sys

CWD = Path(__file__).resolve()

sys.path.append(str(CWD.parents[1]))  # cgmcore/ dir

from file_utils import copy_dir  # noqa: E402

REPO_DIR = str(CWD.parents[4].absolute())


@pytest.fixture
def temp_common_dir():
    temp_common_dir = Path(CWD.parent / "temp_common")
    yield temp_common_dir
    try:
        shutil.rmtree(temp_common_dir)
    except OSError:
        pass


@pytest.fixture
def empty_dir():
    empty_dir = Path(CWD.parent / "copy_empty")
    empty_dir.mkdir(parents=True, exist_ok=True)
    yield empty_dir
    try:
        shutil.rmtree(empty_dir)
    except OSError:
        pass


@pytest.fixture
def temp_empty_dir():
    temp_empty_dir = Path(CWD.parent / "temp_empty_dir")
    yield temp_empty_dir
    try:
        shutil.rmtree(temp_empty_dir)
    except OSError:
        pass


def test_copy_dir(temp_common_dir):
    common_dir_path = Path(REPO_DIR + "/src/common")
    copy_dir(src=common_dir_path, tgt=temp_common_dir, glob_pattern='*/*.py', should_touch_init=True)
    assert temp_common_dir.is_dir(), 'The temp_common_dir does not exist. Did copy_dir fail?'


def test_copy_empty_dir(empty_dir, temp_empty_dir):
    copy_dir(src=empty_dir, tgt=temp_empty_dir, glob_pattern='*/*.py', should_touch_init=False)
    assert temp_empty_dir.is_dir(), 'The temp_empty_dir does not exist. Did copy_dir fail for an empty directory?'
