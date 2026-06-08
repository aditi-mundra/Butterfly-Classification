import numpy as np

def normalize_images(images):
    return images.astype("float32") / 255.0

def get_dataset_info(images, labels):
    return {
        "num_images": len(images),
        "image_shape": images.shape,
        "num_classes": len(np.unique(labels))
    }