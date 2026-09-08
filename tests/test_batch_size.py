import numpy as np
import pytest
from glass_box_umap import ParametricUMAP, recommend_batch_size
from glass_box_umap.batch_size import (
    BatchSizeCandidate,
    _adaptive_batch_sizes,
    _apply_quality_gate,
    _choose_recommendation,
)


def test_adaptive_candidates_preserve_step_floor():
    sizes = _adaptive_batch_sizes(507_848, 10_000, 20, "quick")
    assert sizes == [5_000, 10_000, 20_000, 24_576]
    assert all(np.ceil(507_848 / size) >= 20 for size in sizes)


def test_selection_prefers_smallest_batch_on_throughput_plateau():
    candidates = [
        BatchSizeCandidate(10_000, 50, "ok", 500_000, epoch_losses=(0.5, 0.4)),
        BatchSizeCandidate(20_000, 25, "ok", 610_000, epoch_losses=(0.5, 0.41)),
        BatchSizeCandidate(24_000, 21, "ok", 620_000, epoch_losses=(0.5, 0.41)),
    ]
    _apply_quality_gate(
        candidates,
        candidates[0],
        mode="quick",
        effective_minimum_steps=20,
        loss_tolerance=0.05,
    )
    assert _choose_recommendation(candidates, 0.05).batch_size == 20_000


def test_quality_gate_rejects_loss_degradation_and_too_few_steps():
    candidates = [
        BatchSizeCandidate(10, 30, "ok", 100.0, epoch_losses=(0.5, 0.4)),
        BatchSizeCandidate(20, 20, "ok", 200.0, epoch_losses=(0.5, 0.43)),
        BatchSizeCandidate(30, 10, "ok", 300.0, epoch_losses=(0.5, 0.4)),
    ]
    _apply_quality_gate(
        candidates,
        candidates[0],
        mode="quick",
        effective_minimum_steps=20,
        loss_tolerance=0.05,
    )
    assert candidates[0].eligible
    assert not candidates[1].eligible
    assert candidates[1].rejection_reason is not None
    assert "loss curve" in candidates[1].rejection_reason
    assert not candidates[2].eligible
    assert candidates[2].rejection_reason is not None
    assert "steps/epoch" in candidates[2].rejection_reason


def test_recommender_runs_without_mutating_model():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 6)).astype(np.float32)
    model = ParametricUMAP(
        n_neighbors=3,
        batch_size=8,
        epochs=1,
        random_state=42,
        quiet=True,
    ).to("cpu")

    result = recommend_batch_size(
        X,
        model,
        mode="quick",
        batch_sizes=[4, 8],
        minimum_steps_per_epoch=5,
    )

    assert result.recommended_batch_size in {4, 8}
    assert result.graph_edges > 0
    assert result.summary().startswith("Recommended batch_size=")
    assert result.to_dict()["recommended_batch_size"] == result.recommended_batch_size
    assert model.batch_size == 8
    assert model._model is None


@pytest.mark.parametrize("mode", ["invalid", "QUICK"])
def test_invalid_mode(mode):
    model = ParametricUMAP().to("cpu")
    with pytest.raises(ValueError, match="mode"):
        recommend_batch_size(np.ones((10, 2), dtype=np.float32), model, mode=mode)
