#!/usr/bin/env python
# coding: utf-8
import sys
from pathlib import Path

from PIL import Image

sys.path.append(str(Path(__file__).parents[1]))

from imgseg.predict import show  # noqa: E402

from maskrcnn_resizeimage import get_mask_information, predict_by_resize  # noqa: E402

import numpy as np  # noqa: E402

IMAGE_FNAME = "rgb_test.jpg"


def test_maskrcnn_resizeimage():
    """ testing maskrcnn on resized images """
    source_path = str(Path(__file__).parent / IMAGE_FNAME)

    # Load Image
    image = Image.open(source_path)

    # Prediction
    segmented_image = predict_by_resize(image, factor=10)  # Applying MaskRCNN

    masks = segmented_image['masks'][0][0]

    # Checking the size of the resized segmented image
    width = len(masks[0])
    height = len(masks)
    assert width == 197
    assert height == 120

    # Checking the pixel values of the segmented image
    assert masks[0][0] == 0  # Background
    assert masks[int(height / 2)][int(width / 2)] == 1  # Child

    # Display the mask region, by default
    blank_img_arr = np.zeros((height, width, 3), np.uint8)
    blank_img = Image.fromarray(blank_img_arr, 'RGB')
    # Displays the mask against black background
    show(blank_img, segmented_image, alpha=0.9999)

    """ Checking for the mask area
        & body coverage in percentage of total image area """
    mask_info = get_mask_information(segmented_image)
    assert mask_info[0] == 5147
    assert mask_info[1] == 21.77
