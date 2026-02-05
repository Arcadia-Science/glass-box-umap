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
    )

    model.fit(mnist_images)
    embedding_before = model.transform(mnist_images)

    save_path = tmp_path / "model.pt"
    model.save(save_path)

    loaded_model = ParametricUMAP.load(save_path)
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
    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        model.transform(data)


def test_reproducibility(mnist_images):
    """Ensure random_state produces identical embeddings."""
    seed = 42

    model_a = ParametricUMAP(random_state=seed, epochs=2)
    model_a.fit(mnist_images)
    emb_a = model_a.transform(mnist_images)

    model_b = ParametricUMAP(random_state=seed, epochs=2)
    model_b.fit(mnist_images)
    emb_b = model_b.transform(mnist_images)

    np.testing.assert_array_equal(emb_a, emb_b)

    # Run third model with a different seed to verify different result.
    model_c = ParametricUMAP(random_state=seed + 1, epochs=2)
    model_c.fit(mnist_images)
    emb_c = model_c.transform(mnist_images)
    assert not np.allclose(emb_a, emb_c)


def test_batched_transform(mnist_images: Tensor):
    """Ensure batched transform produces identical results to standard transform."""
    mnist_images = mnist_images
    n_samples = len(mnist_images)

    model = ParametricUMAP(epochs=1, n_components=2)
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
