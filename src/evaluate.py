import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


def get_predictions(model, X_test):

    predictions = model.predict(X_test)

    return np.argmax(predictions, axis=1)


def get_true_labels(y_test):

    return np.argmax(y_test, axis=1)


def generate_classification_report(
    y_true,
    y_pred,
    class_names
):

    return classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )


def generate_confusion_matrix(
    y_true,
    y_pred
):

    return confusion_matrix(
        y_true,
        y_pred
    )