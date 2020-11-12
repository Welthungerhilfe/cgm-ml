import os
import shlex
import subprocess
from pathlib import Path

import azureml.core
from azureml.core import Workspace
from azureml.core.dataset import Dataset

ws = Workspace.from_config()

class Empty(Exception):
    pass

def mount(src, dst=None):
    """ mount azure dataset in current folder
    e.g. mount('anon_pcd_training', "train")
    """
    if not dst:
        dst = src
    dst = Path(dst)
    sdst = str(dst)
    dataset = Dataset.get_by_name(ws, name=src)

    # check existing
    try:
        for _ in dst.iterdir():
            # existing folder and contains files. possibly already mounted and working.
            return
        raise Empty
    except FileNotFoundError:
        # not mounted
        pass
    except (OSError, Empty):
        # mounted but disconnected so cleanup
        subprocess.run(shlex.split(f"sudo umount -f {sdst}"))
        dst.rmdir()

    # mount
    mnt = dataset.mount(sdst)
    mnt.start()
