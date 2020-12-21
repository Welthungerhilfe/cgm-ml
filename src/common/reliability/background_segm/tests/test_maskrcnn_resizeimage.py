"""Testcases for maskrcnn resize image file."""
import sys
from pathlib import Path

from PIL import Image

sys.path.append(str(Path(__file__).parents[1]))

from maskrcnn_resizeimage import predict_by_resize  # noqa: E402

IMAGE_FNAME = "rgb_test.jpg"


def test_maskrcnn_resizeimage():
    """Testing maskrcnn on resized images."""
    source_path = str(Path(__file__).parent / IMAGE_FNAME)

    # Load Image
    image = Image.open(source_path)

    # Prediction
    segmented_image = predict_by_resize(image, factor=10)  # Applying MaskRCNN  # Caveat: takes more than 7seconds

    mask = segmented_image['masks'][0][0]

    # Check the size of the resized segmented image
    height, width = mask.shape
    assert width > 100
    assert height > 100

    # Check the pixel values of the mask
    assert mask[0][0] == 0  # Background (0) is in the corner
    assert mask[int(height / 2)][int(width / 2)] == 1  # Child (1) is in the middle
