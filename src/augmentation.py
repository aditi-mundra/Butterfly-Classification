from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_training_generator():
    return ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )


def create_validation_generator():
    return ImageDataGenerator()