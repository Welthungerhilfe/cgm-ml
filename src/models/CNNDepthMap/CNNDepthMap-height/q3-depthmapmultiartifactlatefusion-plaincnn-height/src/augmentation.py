import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf

from config import CONFIG, DATA_AUGMENTATION_SAME_PER_CHANNEL, DATA_AUGMENTATION_DIFFERENT_EACH_CHANNEL, DATA_AUGMENTATION_NO


@tf.function(input_signature=[tf.TensorSpec(None, tf.float32),  # (240,180,5)
                              tf.TensorSpec(None, tf.float32),  # (1,)
                              ])
def tf_augment_sample(depthmap, targets):
    depthmap_aug = tf.numpy_function(augment, [depthmap, CONFIG.DATA_AUGMENTATION_MODE], tf.float32)
    depthmap_aug.set_shape((CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, CONFIG.N_ARTIFACTS))
    targets.set_shape((len(CONFIG.TARGET_INDEXES,)))

    return depthmap_aug, targets


def augment(image: np.ndarray, mode=DATA_AUGMENTATION_SAME_PER_CHANNEL) -> np.ndarray:
    assert len(image.shape) == 3, f"image array should have 3 dimensions, but has {len(image.shape)}"
    height, width, n_channels = image.shape
    image = image.astype(np.float32)
    mode = mode.decode("utf-8") if isinstance(mode, bytes) else mode

    if mode == DATA_AUGMENTATION_SAME_PER_CHANNEL:
        # Split channel into separate greyscale images
        # for imgaug this order means: (N, height, width, channels)
        image_reshaped = image.reshape(n_channels, height, width, 1)
        return gen_data_aug_sequence().augment_images(image_reshaped).reshape(height, width, n_channels)

    elif mode == DATA_AUGMENTATION_DIFFERENT_EACH_CHANNEL:
        image_augmented = np.zeros((height, width, n_channels), dtype=np.float32)
        for i in range(n_channels):
            onechannel_img = image[:, :, i]
            image_augmented[:, :, i] = gen_data_aug_sequence().augment_images(onechannel_img).reshape(height, width)
        return image_augmented

    elif mode == DATA_AUGMENTATION_NO:
        return image
    else:
        raise NameError(f"{mode}: unknown data aug mode")


def gen_data_aug_sequence():
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-10, 10)),
        # brightness  # TODO find out if this makes sense for depthmaps (talk to Lubos)
        sometimes(iaa.Multiply((0.95, 1.1))),
        iaa.CropAndPad(percent=(-0.02, 0.02), pad_cval=(-0.1, 0.1)),  # TODO is this useful for regression on photos?
        iaa.GaussianBlur(sigma=(0, 1.0)),
        sometimes(
            iaa.OneOf(
                [
                    iaa.Dropout((0.01, 0.05)),
                    iaa.CoarseDropout((0.01, 0.03), size_percent=(0.05, 0.1)),
                    iaa.AdditiveGaussianNoise(scale=(0.0, 0.1)),
                    iaa.SaltAndPepper((0.001, 0.005)),
                ]
            ),
        ),
    ])
    return seq


def sometimes(aug):
    """Randomly enable/disable some of the augmentations"""
    return iaa.Sometimes(0.5, aug)
