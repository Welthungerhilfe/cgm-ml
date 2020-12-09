import os
import re
from functools import partial
from itertools import groupby, islice
from typing import Iterator, List

import glob2 as glob
import numpy as np

from .model_utils_constants import SAMPLING_STRATEGY_SYSTEMATIC, SAMPLING_STRATEGY_WINDOW

REGEX_PICKLE = re.compile(
    r"pc_(?P<qrcode>[a-zA-Z0-9]+-[a-zA-Z0-9]+)_(?P<unixepoch>\d+)_(?P<code>\d{3})_(?P<idx>\d{3}).p$"
)

def create_samples(qrcode_paths: List[str], CONFIG) -> List[List[str]]:
    samples = []
    for qrcode_path in sorted(qrcode_paths):
        for code in CONFIG.CODES_FOR_POSE_AND_SCANSTEP:
            p = os.path.join(qrcode_path, code)
            new_samples = create_multiartifact_paths(p, CONFIG.N_ARTIFACTS, CONFIG)
            samples.extend(new_samples)
    return samples


def create_multiartifact_paths(qrcode_path: str, n_artifacts: int, CONFIG) -> List[List[str]]:
    """Look at files for 1 qrcode and divide into samples.

    Args:
        qrcode_path: File path of 1 qrcode, e.g. "dataset/scans/1583462470-16tvfmb1d0/100"
        n_artifacts: Desired number of artifacts in one sample

    Returns:
        List of samples, where each sample consists of muliple file paths
    """
    path_with_wildcard = os.path.join(qrcode_path, "*.p")
    list_of_pickle_file_paths = sorted(glob.glob(path_with_wildcard))

    # Split if there are multiple scans on different days
    scans = [list(v) for _unixepoch, v in groupby(list_of_pickle_file_paths, _get_epoch)]

    # Filter to keep scans with enough artifacts
    scans = list(filter(lambda x: len(x) > n_artifacts, scans))

    # Sample artifacts
    if CONFIG.SAMPLING_STRATEGY == SAMPLING_STRATEGY_SYSTEMATIC:
        samples = list(map(partial(sample_systematic_from_artifacts, n_artifacts=n_artifacts), scans))

    if CONFIG.SAMPLING_STRATEGY == SAMPLING_STRATEGY_WINDOW:
        samples = []
        for scan in scans:
            some_samples = list(sample_windows_from_artifacts(scan, n_artifacts=n_artifacts))
            assert len(scan) - n_artifacts + 1 == len(some_samples)
            samples.extend(some_samples)

    return samples


def sample_windows_from_artifacts(artifacts: list, n_artifacts: int) -> Iterator[list]:
    """Sample multiple windows (of length n_artifacts) from list of artifacts

    Args:
        artifacts: e.g. ['001.p', '002.p', '003.p', '004.p', '005.p', '006.p']
        n_artifacts: Desired number of artifacts in one sample

    Returns:
        samples: e.g. [
            ['001.p', '002.p', '003.p', '004.p', '005.p'],
            ['002.p', '003.p', '004.p', '005.p', '006.p'],
        ]
    """
    it = iter(artifacts)
    result = list(islice(it, n_artifacts))
    if len(result) == n_artifacts:
        yield result
    for elem in it:
        result = result[1:] + [elem]
        yield result


def sample_systematic_from_artifacts(artifacts: list, n_artifacts: int) -> list:
    n_artifacts_total = len(artifacts)
    n_skip = n_artifacts_total // n_artifacts  # 20 / 5 = 4
    indexes_to_select = list(range(n_skip // 2, n_artifacts_total, n_skip))[:n_artifacts]
    selected_artifacts = [artifacts[i] for i in indexes_to_select]
    assert len(selected_artifacts) == n_artifacts, str(artifacts)
    return selected_artifacts


def _get_epoch(fname: str) -> str:
    match_result = REGEX_PICKLE.search(fname)
    return match_result.group("unixepoch")


def preprocess_depthmap(depthmap: np.array) -> np.array:
    # TODO here be more code.
    return depthmap.astype("float32")


def preprocess_targets(targets: np.array, targets_indices: list) -> np.array:
    if targets_indices is not None:
        targets = targets[targets_indices]
    return targets.astype("float32")
