import numpy as np
import pytest
from glass_box_umap.components import LayerNormDetached, VmapPReLU
from glass_box_umap.core import DeepPReLUNet, GlassBoxUMAP
from torch import Tensor, nn


@pytest.fixture()
def gb_umap() -> GlassBoxUMAP:
    return GlassBoxUMAP(n_components=2, n_neighbors=5, epochs=1, batch_size=64)


@pytest.fixture()
def gb_umap_pca() -> GlassBoxUMAP:
    return GlassBoxUMAP(n_components=2, n_neighbors=5, epochs=1, batch_size=64, pca_components=10)


def test_parametric_architecture_construction():
    input_size = 50
    hidden_size = 256
    n_components = 2

    net = DeepPReLUNet(
        input_dims=(input_size,),
        n_components=n_components,
        hidden_size=hidden_size,
        n_hidden_layers=5,
    )

    expected = nn.Sequential(
        nn.Linear(input_size, hidden_size, bias=False),
        VmapPReLU(),
        LayerNormDetached(hidden_size),
        nn.Dropout(0.1),
        nn.Linear(hidden_size, hidden_size, bias=False),
        VmapPReLU(),
        LayerNormDetached(hidden_size),
        nn.Dropout(0.1),
        nn.Linear(hidden_size, hidden_size, bias=False),
        VmapPReLU(),
        LayerNormDetached(hidden_size),
        nn.Dropout(0.1),
        nn.Linear(hidden_size, hidden_size, bias=False),
        VmapPReLU(),
        LayerNormDetached(hidden_size),
        nn.Dropout(0.1),
        nn.Linear(hidden_size, hidden_size, bias=False),
        VmapPReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size, n_components, bias=False),
    )

    # Direct comparison doesn't work, so we use `repr` as a shortcut to avoid
    # attribute-by-attribute comparison.
    assert repr(net.model) == repr(expected)


def test_compute_attributions_shape(gb_umap: GlassBoxUMAP, mnist_images: Tensor):
    # Attribution shape is correct.
    X = mnist_images
    gb_umap.fit(X)
    contributions = gb_umap.compute_attributions(X)
    assert contributions.shape == (len(X), 2, X.shape[1])


def test_compute_attributions_shape_pca(gb_umap_pca: GlassBoxUMAP, mnist_images: Tensor):
    # Maps back to raw features when PCA preprocessing is used.
    X = mnist_images
    gb_umap_pca.fit(X)
    contributions = gb_umap_pca.compute_attributions(X)
    assert contributions.shape == (len(X), 2, X.shape[1])


def test_compute_attributions_embeddings(gb_umap: GlassBoxUMAP, mnist_images: Tensor):
    # Contributions sum to the embedding.
    X = mnist_images
    gb_umap.fit(X)
    Z = gb_umap.transform(X)
    contributions = gb_umap.compute_attributions(X)
    Z_reconstructed = contributions.astype(np.float32).sum(axis=-1)
    rel_err = np.abs(Z - Z_reconstructed).max() / (np.abs(Z).max() + 1e-8)
    assert rel_err < 1e-3


def test_compute_attributions_embeddings_pca(gb_umap_pca: GlassBoxUMAP, mnist_images: Tensor):
    # Same property holds when PCA preprocessing is used.
    X = mnist_images
    gb_umap_pca.fit(X)
    Z = gb_umap_pca.transform(X)
    contributions = gb_umap_pca.compute_attributions(X)
    Z_reconstructed = contributions.astype(np.float32).sum(axis=-1)
    rel_err = np.abs(Z - Z_reconstructed).max() / (np.abs(Z).max() + 1e-8)
    assert rel_err < 1e-3
