import sys
import shutil
from pathlib import Path

from evaluate import copy_dir

here = Path(__file__).resolve()  # not sure if this is best practice - find out

sys.path.append(str(here.parents[1])) # noqa

REPO_DIR = str(here.parents[6].absolute())


def test_copy_dir():
    common_dir_path = Path(REPO_DIR + "/src/common")
    temp_common_dir = Path(here.parent / "temp_common")
    copy_dir(src=common_dir_path, tgt=temp_common_dir, glob_pattern='*/*.py', should_touch_init=True)

    assert temp_common_dir.exists()
    try:
        shutil.rmtree(temp_common_dir)
    except OSError as e:
        print("Error: %s : %s" % (temp_common_dir, e.strerror))


#include empty dir
def test_copy_empty_dir():
    #create empty dir
    empty_path = Path(here.parent / "copy_empty")
    empty_path.mkdir(parents=True, exist_ok=True)
    #copy empty dir into temp_empty_dir
    temp_empty_dir = Path(here.parent / "temp_empty_dir")
    copy_dir(src=empty_path, tgt=temp_empty_dir, glob_pattern='*/*.py', should_touch_init=False)

    assert temp_empty_dir.exists()

    try:
        shutil.rmtree(empty_path)
    except OSError as e:
        print("Error: %s : %s" % (empty_path, e.strerror))

    try:
        shutil.rmtree(temp_empty_dir)
    except OSError as e:
        print("Error: %s : %s" % (temp_empty_dir, e.strerror))


def test_always_passes():
    assert True


"""def test_always_fails():
    assert False"""
