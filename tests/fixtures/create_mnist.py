"""Create a small MNIST subset for unit tests.

Downloads MNIST and saves images with labels as PyTorch tensors.

Run from anywhere:
    python tests/fixtures/create_mnist.py --num-samples 200
"""

from pathlib import Path

import torch
import typer
from sklearn.datasets import fetch_openml

app = typer.Typer(pretty_exceptions_enable=False)

OUTPUT_DIR = Path(__file__).parent


@app.command()
def main(num_samples: int = 100) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching MNIST...")
    data, target = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    images = torch.tensor(data[:num_samples, :], dtype=torch.float32)
    labels = torch.tensor(target[:num_samples].astype(int), dtype=torch.int64)

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
