import os
import cv2
import numpy as np

from src.config import IMAGE_SIZE


def load_leeds_dataset(image_dir):
    """
    Load Leeds Butterfly Dataset.

    Labels are extracted from the first
    three digits of the filename.

    Example:

    0010001.png -> class 001
    0020012.png -> class 002
    """

    images = []
    labels = []

    filenames = sorted(os.listdir(image_dir))

    for filename in filenames:

        image_path = os.path.join(image_dir, filename)

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE)

        label = filename[:3]

        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)