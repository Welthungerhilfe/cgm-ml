import logging

import numpy as np
from PIL import Image
from torchvision import transforms

from .coco import NAMES

log = logging.getLogger()

# each class has different color bbox
COLORS = [tuple(c) for c in np.random.randint(0, 255, size=(len(NAMES), 3))]


def predict(image, model, threshold=0.8):
    """Return dict of boxes, scores, labels (class index), names, masks"""
    model.eval()
    transform = transforms.ToTensor()
    image = transform(image)
    image = image.unsqueeze(0)
    out = model(image)[0]

    index = (out["scores"] >= threshold) & (out["labels"] == NAMES.index("person"))
    for k, v in out.items():
        out[k] = v.detach().cpu().numpy()[index]
    out["names"] = [NAMES[i] for i in out["labels"]]
    return out


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image """
    image = np.array(image)
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
            image[:, :, c],
        )
    return Image.fromarray(image)


def show(image, out, alpha=0.9999):
    """Display image with boxes, scores, names, masks """
    image = image.copy()
    masks = out.get("masks", [])

    for mask in masks:
        # mask fill. random color to split items
        mask = mask[0].round().astype(np.uint8)
        color = tuple([255, 255, 255])
        image = apply_mask(image, mask, color, alpha)

    return image
