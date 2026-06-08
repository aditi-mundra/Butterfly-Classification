import numpy as np

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def normalize_images(images):
    return images.astype("float32") / 255.0


def get_dataset_info(images, labels):
    return {
        "num_images": len(images),
        "image_shape": images.shape,
        "num_classes": len(np.unique(labels))
    }


def encode_labels(labels):
    encoder = LabelEncoder()

    encoded = encoder.fit_transform(labels)

    return encoded, encoder


def one_hot_encode(labels):
    return to_categorical(labels)


def create_train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42
):

    return train_test_split(
        images,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )