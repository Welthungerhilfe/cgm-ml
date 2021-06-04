"""To identify the child from the RGB images.

Use the pytorch Mask R-CNN Resnet50 library to identify the child
and then using the mask, applied binary image-segmentation to
represent the child pixel as '1' and background pixel as '0'
Further, calculating the mask area and the percentage of
body pixels to total image pixels
"""
import time
import logging
import logging.config

import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn

from imgseg.predict import predict

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d')

model = maskrcnn_resnet50_fpn(pretrained=True)


def predict_by_resize(image, factor=10):
    """Applied MaskRCNN on downscaled image, by default the factor is 10x."""
    logging.info("Resizing image by %d x", factor)
    newsize = (int(image.size[0] / factor), int(image.size[1] / factor))
    logging.info("Resized Dimension %d", newsize)
    start_time = time.time()
    out = predict(image.resize(newsize), model)
    logging.info("Time: %s s", time.time() - start_time)

    # Binary Image Segmentation
    threshold = 0.5
    masks = out['masks'][0][0]
    masks = masks > threshold
    out['masks'][0][0] = masks.astype(int)

    return out


def get_mask_information(segmented_image):
    """Return the mask information."""
    width = len(segmented_image['masks'][0][0][0])
    height = len(segmented_image['masks'][0][0])

    # Get the masked area
    mask_area = int(np.reshape(
        segmented_image['masks'],
        (-1, segmented_image['masks'].shape[-1])).astype(np.float32).sum())

    # Get mask stats like percentage of body coverage of total area & mask area
    perc_body_covered = (mask_area * 100) / (width * height)
    perc_body_covered = round(perc_body_covered, 2)
    logging.info("Mask Area: %.2f px", mask_area)
    logging.info("Percentage of body pixels to total img pixels: %.2f %", perc_body_covered)
    return mask_area, perc_body_covered
