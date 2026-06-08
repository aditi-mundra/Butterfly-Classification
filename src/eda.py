import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def plot_class_distribution(labels):

    counts = Counter(labels)

    plt.figure(figsize=(12, 5))

    sns.barplot(
        x=list(counts.keys()),
        y=list(counts.values())
    )

    plt.title("Class Distribution")
    plt.xlabel("Butterfly Class")
    plt.ylabel("Number of Images")

    plt.tight_layout()
    plt.show()

def show_sample_images(images, labels, n=9):

    plt.figure(figsize=(10,10))

    indices = random.sample(
        range(len(images)),
        n
    )

    for i, idx in enumerate(indices):

        plt.subplot(3, 3, i + 1)

        plt.imshow(images[idx])

        plt.title(labels[idx])

        plt.axis("off")

    plt.tight_layout()

    plt.show()

def save_current_figure(path):

    plt.savefig(
        path,
        bbox_inches="tight",
        dpi=300
    )