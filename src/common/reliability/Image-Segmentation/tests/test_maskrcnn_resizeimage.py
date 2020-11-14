#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[1]))

from maskrcnn_resizeimage import processImage  # noqa: E402

IMAGE_FNAME = "rgb_test.jpg"

def test_maskrcnn_resizeimage():
    source_path = str(Path(__file__).parent / IMAGE_FNAME)
    assert processImage(source_path, target_path)
