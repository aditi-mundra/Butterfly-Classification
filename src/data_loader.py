import os
import cv2
import numpy as np

from src.config import IMAGE_SIZE


def load_images_from_directory(directory):
    """
    Loads images and labels from a dataset directory.

    Expected structure:

    dataset/
    ├── class_1/
    ├── class_2/
    └── class_3/
    """

    images = []
    labels = []

    classes = sorted(os.listdir(directory))

    for label_idx, class_name in enumerate(classes):

        class_path = os.path.join(directory, class_name)

        if not os.path.isdir(class_path):
            continue

        for image_name in os.listdir(class_path):

            image_path = os.path.join(class_path, image_name)

            image = cv2.imread(image_path)

            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMAGE_SIZE)

            images.append(image)
            labels.append(label_idx)

    return np.array(images), np.array(labels), classes