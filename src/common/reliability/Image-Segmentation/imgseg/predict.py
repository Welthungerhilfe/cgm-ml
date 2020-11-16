import logging

from PIL import Image, ImageFont

import numpy as np

from torchvision import transforms

# coco class names
from . import coco

log = logging.getLogger()

# each class has different color bbox
COLORS = [tuple(c) for c in np.random.randint(0, 255, size=(len(coco.names), 3))]

# font to show class names
fontfile = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
font = ImageFont.truetype(fontfile, 12)


def predict(image, model, threshold=0.8):
    """ return dict of boxes, scores, labels (class index), names, masks """
    model.eval()
    transform = transforms.ToTensor()
    image = transform(image)
    image = image.unsqueeze(0)
    out = model(image)[0]

    index = (out["scores"] >= threshold) & (out["labels"] == coco.names.index("person"))
    for k, v in out.items():
        out[k] = v.detach().cpu().numpy()[index]
    out["names"] = [coco.names[i] for i in out["labels"]]
    return out


def apply_mask(image, mask, color, alpha=0.5):
    """ spply the given mask to the image """
    image = np.array(image)
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
            image[:, :, c],
        )
    return Image.fromarray(image)


def show(image, out, alpha=0.9999):
    """ display image with boxes, scores, names, masks """
    image = image.copy()
    masks = out.get("masks", [])

    for i, mask in enumerate(masks):
        # mask fill. random color to split items
        mask = mask[0].round().astype(np.uint8)
        color = tuple([255, 255, 255])
        image = apply_mask(image, mask, color, alpha)

    return image
