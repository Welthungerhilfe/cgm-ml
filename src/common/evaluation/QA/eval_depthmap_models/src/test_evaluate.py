import sys
from pathlib import Path

here = Path(__file__).resolve()

sys.path.append(str(here.parents[1])) # noqa

#from evaluation.QA.eval_depthmap_models.evaluate import copy_dir 
#from evaluate import copy_dir 
REPO_DIR = str(here.parents[6].absolute())

print("Before")
print(REPO_DIR)

""" def test_copy_dir():
    print("Inside")
    common_dir_path = REPO_DIR / "src/common"
    temp_common_dir = Path(__file__).parent / "temp_common"
    copy_dir(src=common_dir_path, tgt=temp_common_dir, glob_pattern='*/*.py', should_touch_init=True)
    if temp_common_dir.exists():
        print("Temp folder exists")

test_copy_dir() """

""" def copy_dir(src: Path, tgt: Path, glob_pattern: str, should_touch_init: bool = False):
    logging.info("Creating temp folder")
    if tgt.exists():
        shutil.rmtree(tgt)
    tgt.mkdir(parents=True, exist_ok=True)
    if should_touch_init:
        (tgt / '__init__.py').touch(exist_ok=False)

    paths_to_copy = list(src.glob(glob_pattern))
    logging.info(f"Copying to {tgt} the following files: {str(paths_to_copy)}")
    for p in paths_to_copy:
        destpath = tgt / p.relative_to(src)
        destpath.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, destpath) """

# so wird sie aufgerufen
    # Copy common into the temp folder
    #common_dir_path = REPO_DIR / "src/common"
    #temp_common_dir = Path(__file__).parent / "temp_common"
    #copy_dir(src=common_dir_path, tgt=temp_common_dir, glob_pattern='*/*.py', should_touch_init=True)