from pathlib import Path

import numpy as np
import pytest
import torch
from glass_box_umap.parametric_umap import ParametricUMAP
from torch import Tensor


def test_roundtrip_serialization_default_encoder(mnist_images: Tensor, tmp_path: Path):
    model = ParametricUMAP(
        encoder_name="default",
        n_neighbors=5,
        epochs=1,
        n_components=2,
    ).to("cpu")

    model.fit(mnist_images)
    embedding_before = model.transform(mnist_images)

    save_path = tmp_path / "model.pt"
    model.save(save_path)

    loaded_model = ParametricUMAP.load(save_path).to("cpu")
    embedding_after = loaded_model.transform(mnist_images)

    assert loaded_model.encoder_name == "default"
    assert loaded_model.n_components == 2
    np.testing.assert_array_almost_equal(embedding_before, embedding_after)


def test_fresh_refit():
    """Ensure model automatically rebuilds if fit is called twice."""
    model = ParametricUMAP(epochs=1, n_components=3)

    data = torch.randn(100, 10)
    model.fit(data)
    original_model = model._model
    assert original_model is not None
    assert original_model.input_dims == (10,)

    # New data has a different dimension.
    new_data = torch.randn(100, 8)
    model.fit(new_data)

    # The original model was replaced.
    assert model._model is not None
    assert model._model is not original_model
    assert model._model.input_dims == (8,)

    # Inference works on the new data.
    output = model.transform(new_data)
    assert output.shape == (100, 3)

    # Inference fails on the original data due to shape mismatch.
    with pytest.raises(ValueError, match="could not be broadcast"):
        model.transform(data)


def test_reproducibility(mnist_images):
    """Ensure random_state produces identical embeddings."""
    seed = 42

    # Pin training to CPU: MPS has non-deterministic kernels that `random_state`
    # (via `pl.seed_everything`) cannot control, which makes this test flaky on
    # Apple Silicon. CPU provides a deterministic substrate to validate the
    # seeding contract itself.
    model_a = ParametricUMAP(random_state=seed, epochs=2).to("cpu")
    model_a.fit(mnist_images)
    emb_a = model_a.transform(mnist_images)

    model_b = ParametricUMAP(random_state=seed, epochs=2).to("cpu")
    model_b.fit(mnist_images)
    emb_b = model_b.transform(mnist_images)

    np.testing.assert_array_equal(emb_a, emb_b)

    # Run third model with a different seed to verify different result.
    model_c = ParametricUMAP(random_state=seed + 1, epochs=2).to("cpu")
    model_c.fit(mnist_images)
    emb_c = model_c.transform(mnist_images)
    assert not np.allclose(emb_a, emb_c)


def test_batched_transform(mnist_images: Tensor):
    """Ensure batched transform produces identical results to standard transform."""
    mnist_images = mnist_images
    n_samples = len(mnist_images)

    model = ParametricUMAP(epochs=1, n_components=2).to("cpu")
    model.fit(mnist_images)

    emb_single_batch = model.transform(mnist_images, batch_size=10000)
    emb_multi_batch = model.transform(mnist_images, batch_size=17)

    assert emb_multi_batch.shape == (n_samples, 2)

    np.testing.assert_array_almost_equal(
        emb_single_batch,
        emb_multi_batch,
        decimal=4,
        err_msg="Batched inference results diverged from standard inference",
    )


def test_pca_preprocessing(mnist_images: Tensor):
    """Ensure PCA preprocessing reduces input dimensions before training."""
    n_pcs = 20
    model = ParametricUMAP(pca_components=n_pcs, epochs=1, n_components=2)
    model.fit(mnist_images)

    assert model._pca is not None
    assert model._pca.n_components == n_pcs
    assert model._fitted_model.input_dims == (n_pcs,)

    embedding = model.transform(mnist_images)
    assert embedding.shape == (len(mnist_images), 2)


def test_pca_disabled_by_default(mnist_images: Tensor):
    """Ensure PCA is not applied when pca_components is None."""
    model = ParametricUMAP(epochs=1, n_components=2)
    model.fit(mnist_images)

    assert model._pca is None
    assert model._fitted_model.input_dims == (mnist_images.shape[1],)


def test_pca_roundtrip_serialization(mnist_images: Tensor, tmp_path: Path):
    """Ensure PCA state is preserved through save/load."""
    n_pcs = 15
    model = ParametricUMAP(pca_components=n_pcs, epochs=1, n_components=2).to("cpu")
    model.fit(mnist_images)
    embedding_before = model.transform(mnist_images)

    save_path = tmp_path / "model_with_pca.pt"
    model.save(save_path)

    loaded_model = ParametricUMAP.load(save_path).to("cpu")

    assert model._pca is not None
    assert loaded_model._pca is not None
    assert loaded_model.pca_components == n_pcs
    np.testing.assert_array_almost_equal(
        model._pca.components_,
        loaded_model._pca.components_,
    )

    embedding_after = loaded_model.transform(mnist_images)
    np.testing.assert_array_almost_equal(embedding_before, embedding_after, decimal=4)


def test_mean_roundtrip_serialization(mnist_images: Tensor, tmp_path: Path):
    """Ensure mean is preserved through save/load."""
    model = ParametricUMAP(epochs=1, n_components=2).to("cpu")
    model.fit(mnist_images)
    embedding_before = model.transform(mnist_images)

    save_path = tmp_path / "model.pt"
    model.save(save_path)

    loaded_model = ParametricUMAP.load(save_path).to("cpu")

    assert model._mean is not None
    assert loaded_model._mean is not None
    np.testing.assert_array_almost_equal(model._mean, loaded_model._mean)

    embedding_after = loaded_model.transform(mnist_images)
    np.testing.assert_array_almost_equal(embedding_before, embedding_after, decimal=4)


def test_conv_encoder_fits_4d_image_input():
    """Ensure encoder_name='default_conv' fits and transforms 4D image input.

    This is the only end-to-end test that exercises a multi-dimensional
    input path through ParametricUMAP.fit. It guards both the registry
    wiring (``encoder_name='default_conv'`` instantiates ConvEncoder with
    the right ``input_dims`` and threads through fit/transform) and the
    fit-time flatten in ``get_umap_graph`` that lets NNDescent see a 2D
    view while the encoder keeps the original 4D shape. The isolated
    encoder tests in ``test_models.py`` do not cover either path.
    """
    n_samples = 32
    input_dims = (1, 8, 8)
    n_components = 2

    X = torch.randn(n_samples, *input_dims)

    model = ParametricUMAP(
        encoder_name="default_conv",
        n_neighbors=5,
        epochs=2,
        n_components=n_components,
    )
    model.fit(X)
    embedding = model.transform(X)

    assert embedding.shape == (n_samples, n_components)


def test_quiet_suppresses_output(mnist_images: Tensor, capfd):
    """Ensure quiet=True silences all stdout/stderr output during fit."""
    model = ParametricUMAP(epochs=1, n_neighbors=5, quiet=True)
    model.fit(mnist_images)
    captured = capfd.readouterr()
    assert captured.out == "", f"Unexpected stdout: {captured.out}"
    assert captured.err == "", f"Unexpected stderr: {captured.err}"

    # When quiet is not set, output is captured.
    model = ParametricUMAP(epochs=1, n_neighbors=5)
    model.fit(mnist_images)
    captured = capfd.readouterr()
    assert captured.out != ""
