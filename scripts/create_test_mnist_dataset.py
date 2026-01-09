"""Create a small MNIST subset for unit tests.

Downloads MNIST and saves images with labels as PyTorch tensors.

Run from the repository root:
    python scripts/create_test_mnist.py --num-samples 12000
"""

from pathlib import Path

import torch
import typer
from sklearn.datasets import fetch_openml

app = typer.Typer(pretty_exceptions_enable=False)

OUTPUT_DIR = Path("tests/fixtures")


@app.command()
def main(num_samples: int = 200) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching MNIST...")
    mnist = fetch_openml("mnist_784", version=1)

    images = torch.tensor(mnist.data.values[:num_samples, :], dtype=torch.float32)
    labels = torch.tensor(mnist.target.values[:num_samples].astype(int), dtype=torch.int64)

    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

    images_path = OUTPUT_DIR / "mnist_images.pt"
    labels_path = OUTPUT_DIR / "mnist_labels.pt"

    torch.save(images, images_path)
    torch.save(labels, labels_path)

    print(f"Saved images to {images_path}")
    print(f"Saved labels to {labels_path}")


if __name__ == "__main__":
    app()
