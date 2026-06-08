from tensorflow.keras.optimizers import Adam


def compile_model(model):

    model.compile(
        optimizer=Adam(
            learning_rate=1e-4
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model