from pathlib import Path

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)


def get_callbacks():

    model_dir = Path("../outputs/models")
    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath=model_dir / "best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    return [early_stop, checkpoint]