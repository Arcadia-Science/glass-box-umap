import numpy as np
import pytest
from glass_box_umap.jacobian import reduce_contributions
from glass_box_umap.plotting.bokeh._data import (
    BarViews,
    TopFeatures,
    compute_bar_views,
    precompute_top_features,
    select_top_features,
    validate_shapes,
)
from numpy.typing import NDArray

N_SAMPLES = 50
N_FEATURES = 12


def _make_inputs(
    seed: int = 0,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    list[str],
    NDArray[np.integer],
]:
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((N_SAMPLES, 2)).astype(np.float32)
    contributions = rng.standard_normal((N_SAMPLES, 2, N_FEATURES)).astype(np.float32)
    feature_names = [f"g_{i}" for i in range(N_FEATURES)]
    group_names = rng.integers(0, 4, size=N_SAMPLES)
    return Z, contributions, feature_names, group_names


def test_validate_shapes_happy_path():
    Z, contributions, feature_names, group_names = _make_inputs()
    validate_shapes(Z, contributions, feature_names=feature_names, group_names=group_names)


def test_validate_shapes_rejects_non_2d_Z():
    Z, contributions, *_ = _make_inputs()
    with pytest.raises(ValueError, match=r"Z must have shape"):
        validate_shapes(Z[:, :1], contributions)


def test_validate_shapes_rejects_sample_count_mismatch():
    Z, contributions, *_ = _make_inputs()
    with pytest.raises(ValueError, match=r"contributions\.shape\[0\]"):
        validate_shapes(Z, contributions[:10])


def test_validate_shapes_rejects_wrong_component_axis():
    Z, contributions, *_ = _make_inputs()
    with pytest.raises(ValueError, match=r"contributions\.shape\[1\] must be 2"):
        validate_shapes(Z, contributions[:, :1, :])


def test_validate_shapes_rejects_zero_features():
    Z, contributions, *_ = _make_inputs()
    with pytest.raises(ValueError, match=r"at least one feature"):
        validate_shapes(Z, contributions[:, :, :0])


def test_validate_shapes_rejects_wrong_contributions_ndim():
    Z, *_ = _make_inputs()
    bad = np.zeros((N_SAMPLES, 2), dtype=np.float32)
    with pytest.raises(ValueError, match=r"3 dimensions"):
        validate_shapes(Z, bad)


def test_validate_shapes_rejects_bad_feature_names_length():
    Z, contributions, *_ = _make_inputs()
    with pytest.raises(ValueError, match=r"feature_names has length"):
        validate_shapes(Z, contributions, feature_names=["only_one"])


def test_validate_shapes_rejects_bad_group_names_length():
    Z, contributions, _, group_names = _make_inputs()
    with pytest.raises(ValueError, match=r"group_names has length"):
        validate_shapes(Z, contributions, group_names=group_names[:10])


def test_select_top_features_returns_sorted_pool():
    _, contributions, feature_names, _ = _make_inputs()
    result = select_top_features(
        contributions,
        feature_names,
        top_k_global=5,
        top_k_display=3,
    )
    assert isinstance(result, TopFeatures)
    assert result.n_kept == 5
    assert result.display_k == 3
    assert len(result.kept_names) == 5
    assert result.keep_idx.shape == (5,)
    assert result.reduced.shape == (N_SAMPLES, N_FEATURES)
    assert result.kept_names == [feature_names[i] for i in result.keep_idx]


def test_select_top_features_clips_to_available_features():
    _, contributions, feature_names, _ = _make_inputs()
    result = select_top_features(
        contributions,
        feature_names,
        top_k_global=10_000,
        top_k_display=10_000,
    )
    assert result.n_kept == N_FEATURES
    assert result.display_k == N_FEATURES


def test_select_top_features_synthesizes_names_when_none():
    _, contributions, *_ = _make_inputs()
    result = select_top_features(
        contributions,
        None,
        top_k_global=3,
        top_k_display=3,
    )
    for name in result.kept_names:
        assert name.startswith("Feature ")


def test_compute_bar_views_matches_source():
    _, contributions, feature_names, _ = _make_inputs()
    top = select_top_features(contributions, feature_names, top_k_global=5, top_k_display=3)
    views = compute_bar_views(contributions, top)
    assert isinstance(views, BarViews)
    expected_l2 = reduce_contributions(contributions, "l2")[:, top.keep_idx].astype(np.float32)
    np.testing.assert_array_equal(views.l2, expected_l2)
    np.testing.assert_array_equal(views.d0, contributions[:, 0, top.keep_idx].astype(np.float32))
    np.testing.assert_array_equal(views.d1, contributions[:, 1, top.keep_idx].astype(np.float32))
    assert (views.l2 >= 0).all()


def test_precompute_top_features_ranks_by_frequency():
    kept_l2 = np.array(
        [
            [3.0, 1.0, 2.0],
            [3.0, 1.0, 0.5],
            [0.5, 0.5, 5.0],
            [4.0, 1.0, 2.0],
            [0.1, 0.2, 0.3],
        ],
        dtype=np.float32,
    )
    kept_names = ["alpha", "beta", "gamma"]
    names_by_rank, sample_rank, top_kept_idx = precompute_top_features(kept_l2, kept_names)
    assert names_by_rank == ["alpha", "gamma"]
    np.testing.assert_array_equal(sample_rank, np.array([0, 0, 1, 0, 1]))
    np.testing.assert_array_equal(top_kept_idx, np.array([0, 0, 2, 0, 2]))
    assert sample_rank.max() < len(names_by_rank)


def test_precompute_top_features_handles_single_winner():
    kept_l2 = np.array([[5.0, 1.0], [4.0, 0.5], [3.0, 2.0]], dtype=np.float32)
    names_by_rank, sample_rank, top_kept_idx = precompute_top_features(kept_l2, ["only", "loser"])
    assert names_by_rank == ["only"]
    np.testing.assert_array_equal(sample_rank, np.zeros(3, dtype=sample_rank.dtype))
    np.testing.assert_array_equal(top_kept_idx, np.zeros(3, dtype=top_kept_idx.dtype))
