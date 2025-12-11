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
import typer
from glass_box_umap.glass_box_umap import GlassBoxUMAP
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

app = typer.Typer(pretty_exceptions_enable=False)


def load_and_preprocess_mnist(n_pcs: int, subset: int = 4000) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Fetches MNIST data and performs PCA preprocessing."""
    print("Fetching MNIST...")
    mnist = fetch_openml("mnist_784", version=1)
    data_values = mnist.data.values[:subset, :]
    target_values = mnist.target.values[:subset]

    pca = PCA(n_components=n_pcs)
    pca.fit(data_values)

    mnist_pca = pca.transform(data_values)
    mnist_pca_centered = (mnist_pca.T - mnist_pca.mean(axis=1)).T
    return mnist_pca_centered, target_values, pca


@app.command()
def main(
    output_dir: Path = typer.Option(Path("output"), help="Directory for all output files"),
    n_fits: int = typer.Option(1, help="Number of UMAP fits"),
    n_pcs: int = typer.Option(25, help="Number of PCA components"),
    epochs: int = typer.Option(100, help="Number of training epochs"),
    random_state: int = typer.Option(42, help="Random seed"),
    hidden_size: int = typer.Option(512, help="Hidden layer size"),
    train: bool = typer.Option(True, help="Train model (False to load from disk)"),
) -> None:
    """Train GlassBoxUMAP on MNIST and save outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path_pattern = str(output_dir / "models" / "umap_{i}.pth")
    (output_dir / "models").mkdir(parents=True, exist_ok=True)

    X_pca_centered, y_target, pca_model = load_and_preprocess_mnist(n_pcs)

    print("Initializing GlassBoxUMAP...")
    reducer = GlassBoxUMAP(
        n_fits=n_fits,
        epochs=epochs if train else 0,
        random_state=random_state,
        input_size=n_pcs,
        hidden_size=hidden_size,
    )

    print("Fitting GlassBoxUMAP...")
    reducer.fit(
        X_pca_centered,
        load_models=not train,
        load_n_fits=n_fits,
        save_models=train,
        model_path_pattern=model_path_pattern,
    )

    print("Running manual Jacobian check...")
    model = reducer._models[0]
    encoder = model.encoder
    device = reducer._device
    encoder.eval().to(device)

    sample_tensor = torch.tensor(X_pca_centered[:1, :].squeeze(), dtype=torch.float32).to(device)

    jac_batch = torch.autograd.functional.jacobian(
        encoder, sample_tensor, vectorize=True, strategy="reverse-mode"
    )

    jacobian_np = jac_batch.T.detach().cpu().numpy().T
    reconstruction = jacobian_np @ X_pca_centered[0, :]
    print("Jacobian Reconstruction:", reconstruction)

    forward_pass = encoder(torch.tensor(X_pca_centered[0, :], dtype=torch.float32).to(device))
    print("Forward Pass:", forward_pass)

    print("Plotting embedding...")
    embedding = reducer._embeddings[0]

    sns.set_style("whitegrid")
    ax = sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=y_target, s=3)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1))

    fig = ax.get_figure()
    embedding_path = output_dir / "embedding.png"
    fig.savefig(embedding_path, bbox_inches="tight")
    print(f"Saved embedding plot to {embedding_path}")

    print("Plotting linear operator...")
    plt.figure()

    linear_operator = (pca_model.components_.T @ jac_batch.T.detach().cpu().numpy())[:, 0].reshape(
        [28, 28]
    )

    plt.imshow(linear_operator, cmap="RdBu")
    linear_op_path = output_dir / "linear_operator.png"
    plt.savefig(linear_op_path)
    print(f"Saved linear operator plot to {linear_op_path}")


if __name__ == "__main__":
    app()
