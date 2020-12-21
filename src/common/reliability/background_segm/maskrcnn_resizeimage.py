"""To identify the child from the RGB images.

Use the pytorch Mask R-CNN Resnet50 library to identify the child
and then using the mask, applied binary image-segmentation to
represent the child pixel as '1' and background pixel as '0'
Further, calculating the mask area and the percentage of
body pixels to total image pixels
"""
import time

import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn

from imgseg.predict import predict

model = maskrcnn_resnet50_fpn(pretrained=True)


def predict_by_resize(image, factor=10):
    """Applied MaskRCNN on downscaled image, by default the factor is 10x."""
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
    print("Mask Area:", mask_area, "px")
    print("Percentage of body pixels to total img pixels:", perc_body_covered, "%")
    return mask_area, perc_body_covered
