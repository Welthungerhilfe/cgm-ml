#!/usr/bin/env python
# coding: utf-8
import time

from imgseg.predict import predict

import numpy as np

from torchvision.models.detection import maskrcnn_resnet50_fpn

model = maskrcnn_resnet50_fpn(pretrained=True)


def predict_by_resize(image, factor=10):
    """ Applies MaskRCNN on downscaled image, by default the factor is 10x """
    print("Resizing image by", factor, "x")
    newsize = (int(image.size[0] / factor), int(image.size[1] / factor))
    print("Resized Dimension", newsize)
    start_time = time.time()
    out = predict(image.resize(newsize), model)
    print("Time: %s s" % (time.time() - start_time))

    # Binary Image Segmentation
    threshold = 0.5
    masks = out['masks'][0][0]
    masks = masks > threshold
    out['masks'][0][0] = masks.astype(int)

    return out


def get_mask_information(segmented_image):
    """ gets the information regarding mask like the mask area
    & body percentage """
    width = len(segmented_image['masks'][0][0][0])
    height = len(segmented_image['masks'][0][0])

    # Getting the masked area
    mask_area = int(np.reshape(
        segmented_image['masks'],
        (-1, segmented_image['masks'].shape[-1])).astype(np.float32).sum())

    # Mask Stats like percentage of body coverage of total area & mask area
    perc_body_covered = (mask_area * 100) / (width * height)
    perc_body_covered = round(perc_body_covered, 2)
    print("Mask Area:", mask_area, "px")
    print(
        "Percentage of body pixels to total img pixels:",
        perc_body_covered, "%")
    return (mask_area, perc_body_covered)
