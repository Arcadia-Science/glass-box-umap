from pathlib import Path

import numpy as np
from glass_box_umap.parametric_umap import ParametricUMAP
from torch import Tensor


def test_roundtrip_serialization_default_encoder(mnist_images: Tensor, tmp_path: Path):
    model = ParametricUMAP.create(
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

    assert loaded_model.umap_config.encoder_name == "default"
    assert loaded_model.umap_config.n_components == 2
    np.testing.assert_array_almost_equal(embedding_before, embedding_after)
