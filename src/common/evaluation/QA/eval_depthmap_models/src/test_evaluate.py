import sys
from pathlib import Path

from evaluate import copy_dir 

here = Path(__file__).resolve()

sys.path.append(str(here.parents[1])) # noqa

REPO_DIR = str(here.parents[6].absolute())


def test_copy_dir():
    common_dir_path = Path(REPO_DIR + "/src/common")
    print("common_dir_path: ", common_dir_path)
    temp_common_dir = Path(here.parent / "temp_common")
    print("temp_common_dir: ", temp_common_dir)
    copy_dir(src=common_dir_path, tgt=temp_common_dir, glob_pattern='*/*.py', should_touch_init=True)
    if temp_common_dir.exists():
        print("copying successful, temp_common folder was created.")

test_copy_dir()