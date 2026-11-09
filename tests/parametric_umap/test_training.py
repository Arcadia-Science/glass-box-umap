"""The plain PyTorch trainer: same weights as the Lightning trainer, chosen by ``trainer``."""

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from glass_box_umap import GlassBoxUMAP
from glass_box_umap.parametric_umap import ParametricUMAP
from glass_box_umap.parametric_umap.data import UMAPDataset
from glass_box_umap.parametric_umap.graph import get_umap_graph
from glass_box_umap.parametric_umap.lightning import UMAPLightningModule
from glass_box_umap.parametric_umap.loss import find_ab_params
from glass_box_umap.parametric_umap.module import UMAPModule
from glass_box_umap.parametric_umap.training import seed_everything
from glass_box_umap.utils import lightning_training_device


@pytest.fixture
def blobs() -> np.ndarray:
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=4.0, size=(4, 7))
    return (centers[rng.integers(4, size=180)] + rng.normal(size=(180, 7))).astype(np.float32)


def _fit(X: np.ndarray | torch.Tensor, **kwargs) -> ParametricUMAP:
    defaults = dict(random_state=0, epochs=3, batch_size=256, n_neighbors=8, quiet=True)
    return ParametricUMAP(**{**defaults, **kwargs}).to("cpu").fit(X)


def _assert_same_weights(a: ParametricUMAP, b: ParametricUMAP) -> None:
    sa, sb = a._fitted_model.state_dict(), b._fitted_model.state_dict()
    assert sa.keys() == sb.keys()
    for key in sa:
        assert torch.equal(sa[key], sb[key]), key


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        # Several batches per epoch, the last one smaller.
        {"batch_size": 100},
        {"batch_size": 100, "num_batches": 2},
        {"restore_best_weights": False},
        {"pca_components": 4},
        # With workers, Lightning's DataLoader keeps its iterator between epochs and draws
        # its seed once instead of every epoch.
        {"num_workers": 2},
    ],
)
def test_torch_trainer_matches_lightning_trainer(blobs: np.ndarray, kwargs: dict) -> None:
    fast = _fit(blobs, trainer="torch", **kwargs)
    reference = _fit(blobs, trainer="lightning", **kwargs)
    _assert_same_weights(fast, reference)
    np.testing.assert_array_equal(fast.transform(blobs), reference.transform(blobs))


def test_torch_trainer_matches_lightning_trainer_on_images() -> None:
    X = torch.randn(40, 1, 8, 8, generator=torch.Generator().manual_seed(0))
    kwargs = dict(encoder_name="default_conv", n_neighbors=5, batch_size=64)
    _assert_same_weights(_fit(X, trainer="torch", **kwargs), _fit(X, trainer="lightning", **kwargs))


def _accelerators() -> list[str]:
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.mark.skipif(not _accelerators(), reason="No CUDA or MPS device")
@pytest.mark.parametrize("device", _accelerators())
def test_torch_trainer_on_accelerator(blobs: np.ndarray, device: str) -> None:
    """Batches copied to the device stay intact (an async copy once corrupted them on MPS)."""
    kwargs: dict[str, Any] = dict(
        random_state=0, epochs=3, batch_size=64, n_neighbors=8, quiet=True
    )
    fast = ParametricUMAP(trainer="torch", **kwargs).to(device).fit(blobs)
    reference = ParametricUMAP(trainer="lightning", **kwargs).to(device).fit(blobs)
    Y = fast.transform(blobs)
    assert np.isfinite(Y).all()
    np.testing.assert_allclose(Y, reference.transform(blobs), atol=1e-4)


@pytest.mark.parametrize("disable_mps", ["0", "1"])
@pytest.mark.parametrize("processor", ["arm", "i386"])
def test_training_device_follows_lightnings_mps_fallback(
    monkeypatch: pytest.MonkeyPatch, disable_mps: str, processor: str
) -> None:
    """The torch trainer trains where Lightning would: on the CPU when it rejects MPS."""
    from lightning_fabric.accelerators.mps import MPSAccelerator

    is_available = MPSAccelerator.is_available
    monkeypatch.setenv("DISABLE_MPS", disable_mps)
    monkeypatch.setattr(platform, "processor", lambda: processor)
    is_available.cache_clear()  # type: ignore[attr-defined]
    try:
        expected = "mps" if is_available() else "cpu"
    finally:
        is_available.cache_clear()  # type: ignore[attr-defined]
    assert lightning_training_device(torch.device("mps")) == torch.device(expected)
    assert lightning_training_device(torch.device("cpu")) == torch.device("cpu")


def test_restores_the_best_epoch(blobs: np.ndarray) -> None:
    """The restored weights are the ones the lowest-loss epoch ended with."""
    graph = get_umap_graph(blobs, n_neighbors=8, random_state=0, quiet=True)
    dataset = UMAPDataset(blobs, graph, random_state=0)

    def train(epochs: int, restore: bool) -> tuple[UMAPModule, list[float]]:
        from glass_box_umap.parametric_umap.registry import create_encoder
        from glass_box_umap.parametric_umap.training import train_umap

        seed_everything(0)
        module = UMAPModule(
            lr=5e-2, encoder=create_encoder("default", (7,), 2, {}), input_dims=(7,)
        )
        losses = train_umap(
            module,
            dataset,
            epochs=epochs,
            batch_size=64,
            device=torch.device("cpu"),
            restore_best_weights=restore,
        )
        return module, losses

    restored, losses = train(12, restore=True)
    best = int(np.argmin(losses))
    assert len(losses) == 12
    assert best < 11, "The test needs a run whose loss isn't lowest at the last epoch."
    stopped_at_best, _ = train(best + 1, restore=False)
    for key, value in restored.state_dict().items():
        assert torch.equal(value, stopped_at_best.state_dict()[key]), key


def test_auto_trainer_uses_torch_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    model = ParametricUMAP()
    assert model.trainer == "auto"
    assert model._resolve_trainer() == "torch"


def test_auto_trainer_uses_lightning_for_lightning_options(tmp_path: Path) -> None:
    assert ParametricUMAP(checkpoint_dir=tmp_path)._resolve_trainer() == "lightning"
    with_callback = ParametricUMAP()
    with_callback.extra_callbacks.append(object())  # type: ignore[arg-type]
    assert with_callback._resolve_trainer() == "lightning"


def test_torch_trainer_rejects_lightning_options(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="need the Lightning trainer"):
        ParametricUMAP(trainer="torch", checkpoint_dir=tmp_path)._resolve_trainer()
    with pytest.raises(ValueError, match="trainer must be"):
        ParametricUMAP(trainer="keras")._resolve_trainer()  # type: ignore[arg-type]


def test_checkpoint_dir_still_writes_tensorboard_logs(blobs: np.ndarray, tmp_path: Path) -> None:
    _fit(blobs, checkpoint_dir=tmp_path)
    assert (tmp_path / "checkpoints" / "best.ckpt").exists()
    assert any((tmp_path / "logs").iterdir())


def test_torch_trainer_writes_no_files(
    blobs: np.ndarray, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    _fit(blobs, trainer="torch")
    assert list(tmp_path.iterdir()) == []


def test_find_ab_params_matches_umap_learn() -> None:
    from umap.umap_ import find_ab_params as umap_find_ab_params

    for spread, min_dist in [(1.0, 0.1), (1.0, 0.0), (1.0, 0.5), (2.0, 0.25)]:
        assert find_ab_params(spread, min_dist) == umap_find_ab_params(spread, min_dist)


def test_seed_everything_matches_lightning() -> None:
    import random

    import pytorch_lightning as pl

    pl.seed_everything(7, verbose=False)
    expected = (random.random(), np.random.random(), torch.rand(1).item())
    seed_everything(7)
    assert (random.random(), np.random.random(), torch.rand(1).item()) == expected
    with pytest.raises(ValueError, match="not in bounds"):
        seed_everything(-1)


def test_edge_batch_matches_getitem(blobs: np.ndarray) -> None:
    dataset = UMAPDataset(blobs, get_umap_graph(blobs, n_neighbors=8, random_state=0, quiet=True))
    idx = torch.tensor([0, 5, 5, len(dataset) - 1])
    edges_to, edges_from = dataset.edge_batch(idx)
    items = [dataset[int(i)] for i in idx]
    assert torch.equal(edges_to, torch.stack([a for a, _ in items]))
    assert torch.equal(edges_from, torch.stack([b for _, b in items]))


def test_loads_a_model_saved_by_version_0_1_0(blobs: np.ndarray, tmp_path: Path) -> None:
    """0.1.0 saved the state_dict of a UMAPLightningModule. It loads into UMAPModule."""
    from glass_box_umap.parametric_umap.registry import create_encoder

    encoder = create_encoder("default", (7,), 2, {})
    old = UMAPLightningModule(lr=1e-3, encoder=encoder, input_dims=(7,), min_dist=0.1)
    attrs = dict(
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        n_components=2,
        negative_sample_rate=5,
        repulsion_strength=1.0,
        pca_components=None,
        encoder_name="default",
        encoder_kwargs={},
        lr=1e-3,
        epochs=200,
        batch_size=10_000,
        num_batches=None,
        num_workers=0,
        checkpoint_dir=None,
        restore_best_weights=True,
        random_state=0,
        quiet=True,
    )
    mean = blobs.mean(0)
    torch.save(
        {
            "attrs": attrs,
            "state_dict": old.state_dict(),
            "input_dims": (7,),
            "pca": None,
            "mean": mean,
        },
        tmp_path / "model.pt",
    )

    loaded = GlassBoxUMAP.load(tmp_path / "model.pt").to("cpu")
    assert loaded.trainer == "auto"
    assert loaded.neighbor_search == "nndescent"
    with torch.no_grad():
        expected = encoder.eval()(torch.from_numpy(blobs - mean)).numpy()
    np.testing.assert_array_equal(loaded.transform(blobs), expected)
    C = loaded.compute_contributions(blobs)
    assert np.abs(C.sum(-1) - expected).max() < 1e-4


def _modules_loaded_by(code: str) -> set[str]:
    heavy = (
        "pytorch_lightning",
        "lightning_fabric",
        "tensorboard",
        "umap",
        "pynndescent",
        "numba",
        "sklearn",
    )
    script = (
        f"{code}\n"
        "import sys\n"
        f"print(sorted({{m.split('.')[0] for m in sys.modules}} & set({heavy!r})))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    ).stdout
    return set(eval(out.strip().splitlines()[-1]))


def test_import_loads_no_heavy_dependencies() -> None:
    assert _modules_loaded_by("import glass_box_umap") == set()


def test_fit_with_exact_neighbors_loads_no_heavy_dependencies() -> None:
    code = (
        "import numpy as np\n"
        "from glass_box_umap import GlassBoxUMAP\n"
        "X = np.random.default_rng(0).normal(size=(30, 5)).astype(np.float32)\n"
        "m = GlassBoxUMAP(random_state=0, quiet=True, epochs=2, neighbor_search='exact')\n"
        "m.to('cpu')\n"
        "m.fit(X)\n"
        "m.compute_contributions(X)\n"
    )
    assert _modules_loaded_by(code) == set()


def test_default_fit_does_not_load_lightning() -> None:
    code = (
        "import numpy as np\n"
        "from glass_box_umap import GlassBoxUMAP\n"
        "X = np.random.default_rng(0).normal(size=(30, 5)).astype(np.float32)\n"
        "GlassBoxUMAP(random_state=0, quiet=True, epochs=2).to('cpu').fit(X)\n"
    )
    assert _modules_loaded_by(code).isdisjoint({"pytorch_lightning", "tensorboard"})
