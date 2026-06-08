from src.data_loader import load_leeds_dataset

from src.preprocessing import (
    normalize_images,
    encode_labels,
    one_hot_encode,
    create_train_test_split
)

def prepare_dataset(
    dataset_path="../data/raw/leedsbutterfly/images"
):

    images, labels = load_leeds_dataset(
        dataset_path
    )

    images = normalize_images(images)

    encoded_labels, encoder = encode_labels(
        labels
    )

    categorical_labels = one_hot_encode(
        encoded_labels
    )

    X_train, X_test, y_train, y_test = (
        create_train_test_split(
            images,
            categorical_labels
        )
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        labels,
        encoder
    )