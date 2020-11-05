import os
import logging
from typing import Iterable

import cv2
import face_recognition
import numpy as np


CODES_FRONT_FACING = ("100", "200")
CODES_BACK_FACING = ("102", "202")
CODES_360 = ("101", "201")

RESIZE_FACTOR = 4


def blur_faces_in_file(source_path: str, target_path: str) -> bool:
    """Blur image

    Returns:
        bool: True if blurred otherwise False
    """

    # Read the image.
    assert os.path.exists(source_path), f"{source_path} does not exist"
    rgb_image = cv2.imread(source_path)
    image = rgb_image[:, :, ::-1]  # RGB -> BGR for OpenCV

    # The images are provided in 90degrees turned. Here we rotate 90degress to the right.
    image = np.swapaxes(image, 0, 1)

    # Scale image down for faster prediction.
    small_image = cv2.resize(image, (0, 0), fx=1. / RESIZE_FACTOR, fy=1. / RESIZE_FACTOR)

    # Find face locations.
    face_locations = face_recognition.face_locations(small_image, model="cnn")

    # Check if image should be used.
    if not should_image_be_used(source_path, number_of_faces=len(face_locations)):
        logging.warn(f"{len(face_locations)} face locations found and not blurred for path: {source_path}")
        return False

    # Blur the image.
    for top, right, bottom, left in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= RESIZE_FACTOR
        right *= RESIZE_FACTOR
        bottom *= RESIZE_FACTOR
        left *= RESIZE_FACTOR

        # Extract the region of the image that contains the face.
        face_image = image[top:bottom, left:right]

        # Blur the face image.
        face_image = cv2.GaussianBlur(face_image, ksize=(99, 99), sigmaX=30)

        # Put the blurred face region back into the frame image.
        image[top:bottom, left:right] = face_image

    # Rotate image back.
    image = np.swapaxes(image, 0, 1)

    # Write image to hard drive.
    rgb_image = image[:, :, ::-1]  # BGR -> RGB for OpenCV
    cv2.imwrite(target_path, rgb_image)

    logging.info(f"{len(face_locations)} face locations found and blurred for path: {source_path}")
    return True


def should_image_be_used(source_path: str, number_of_faces: int) -> bool:
    """Determines if an image should be skipped or not."""
    if does_path_belong_to_codes(source_path, CODES_FRONT_FACING):
        return number_of_faces == 1
    elif does_path_belong_to_codes(source_path, CODES_360):
        return True
    elif does_path_belong_to_codes(source_path, CODES_BACK_FACING):
        return True
    else:
        raise NameError(f"{source_path} does not have a correct code")


def does_path_belong_to_codes(path: str, codes: Iterable) -> bool:
    for code in codes:
        if f"_{code}_" in path:
            return True
    return False
