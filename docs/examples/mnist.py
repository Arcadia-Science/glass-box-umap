"""Training and evaluation script for GlassBoxUMAP on MNIST.

This script lives in docs/examples/ but is not yet a rendered notebook example.
It will eventually be promoted to a Jupyter notebook that renders as part of
the documentation. For now, it serves as a development script.

Run from the repository root:

    python docs/examples/mnist.py --output-dir results/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from numpy.typing import NDArray
import typer
from glass_box_umap import GlassBoxUMAP
from sklearn.datasets import fetch_openml

app = typer.Typer(pretty_exceptions_enable=False)


def load_mnist(subset: int = 4000) -> tuple[torch.Tensor, NDArray[np.str_]]:
    """Fetches MNIST data."""
    print("Fetching MNIST...")
    mnist = fetch_openml("mnist_784", version=1)
    data_raw = mnist.data.values[:subset, :].astype(np.float32)
    target_values = mnist.target.values[:subset]

    return torch.from_numpy(data_raw), target_values


@app.command()
def main(
    output_dir: Path = typer.Option(Path("output"), help="Directory for all output files"),
    n_pcs: int = typer.Option(25, help="Number of PCA components"),
    epochs: int = typer.Option(100, help="Number of training epochs"),
    random_state: int = typer.Option(42, help="Random seed"),
    hidden_size: int = typer.Option(512, help="Hidden layer size"),
    train: bool = typer.Option(True, help="Train model (False to load from disk)"),
) -> None:
    """Train GlassBoxUMAP on MNIST and save outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"

    X_raw, y_target = load_mnist()

    if train:
        print("Initializing GlassBoxUMAP...")
        reducer = GlassBoxUMAP(
            pca_components=n_pcs,
            epochs=epochs,
            random_state=random_state,
            encoder_kwargs={"hidden_size": hidden_size},
        )

        print("Fitting GlassBoxUMAP...")
        reducer.fit(X_raw)
        reducer.save(model_path)
        print(f"Saved model to {model_path}")
    else:
        print(f"Loading model from {model_path}...")
        reducer = GlassBoxUMAP.load(model_path)

    print("Computing attributions...")
    feature_contributions, jacobians = reducer.compute_attributions(X=X_raw)

    print("Plotting embedding...")
    embedding = reducer.transform(X_raw)

    sns.set_style("whitegrid")
    ax = sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=y_target, s=3)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1))

    fig = ax.get_figure()
    embedding_path = output_dir / "embedding.png"
    fig.savefig(embedding_path, bbox_inches="tight")
    print(f"Saved embedding plot to {embedding_path}")

    print("Plotting linear operator for first sample...")
    plt.figure()

    jacobian_pixel = jacobians[0].numpy() @ reducer._pca.components_
    linear_operator = jacobian_pixel[0, :].reshape([28, 28])

    plt.imshow(linear_operator, cmap="RdBu")
    linear_op_path = output_dir / "linear_operator.png"
    plt.savefig(linear_op_path)
    print(f"Saved linear operator plot to {linear_op_path}")


if __name__ == "__main__":
    app()
