from tensorflow.keras.applications import Xception

from tensorflow.keras.layers import (
    AveragePooling2D,
    Dropout,
    Flatten,
    Dense
)

from tensorflow.keras.models import Model


def build_xception_model(
    input_shape=(224, 224, 3),
    num_classes=10
):

    base_model = Xception(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output

    x = AveragePooling2D(
        pool_size=(4, 4)
    )(x)

    x = Flatten()(x)

    x = Dense(
        128,
        activation="relu"
    )(x)

    x = Dropout(0.5)(x)

    predictions = Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs=base_model.input,
        outputs=predictions
    )

    return model