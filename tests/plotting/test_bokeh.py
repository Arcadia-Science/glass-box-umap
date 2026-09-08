import numpy as np
import pytest
from bokeh.models import CheckboxGroup, Div, RadioButtonGroup, Select, Slider
from glass_box_umap.jacobian import reduce_contributions
from glass_box_umap.plotting import HierarchyLevel, HierarchySpec, plot_embedding
from glass_box_umap.plotting.bokeh._data import (
    BarViews,
    TopFeatures,
    collapse_position_features,
    compute_bar_views,
    precompute_top_features,
    select_top_features,
    validate_shapes,
)
from glass_box_umap.plotting.bokeh._hierarchy import validate_hierarchies
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


def test_collapse_position_features_uses_max_l2_member_and_keeps_anchor():
    contributions = np.array(
        [
            [[1.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
            [[2.0, 1.0, 0.0], [0.0, 0.0, 0.5]],
        ],
        dtype=np.float32,
    )
    values = np.array([[7.0, 2.0, 5.0], [8.0, 6.0, 4.0]], dtype=np.float32)
    collapsed = collapse_position_features(
        contributions,
        ["ref_region_CDS", "1_ref_region_CDS", "2_ref_region_CDS"],
        values,
    )
    assert collapsed.names == ["ref_region_CDS", "max_ref_region_CDS"]
    np.testing.assert_array_equal(collapsed.contributions[:, :, 0], contributions[:, :, 0])
    np.testing.assert_array_equal(
        collapsed.contributions[:, :, 1],
        np.array([[0.0, 4.0], [1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(collapsed.values[:, 1], np.array([5.0, 6.0]))


def _make_hierarchy(n_samples: int = N_SAMPLES) -> HierarchySpec:
    coarse = np.asarray([f"C{i % 2 + 1}" for i in range(n_samples)])
    fine = np.asarray([f"C{i % 2 + 1}.{i % 4 // 2 + 1}" for i in range(n_samples)])

    def level(name, labels):
        unique = sorted(set(labels))
        return HierarchyLevel(
            name=name,
            labels=labels,
            colors={label: "#336699" for label in unique},
            metadata={
                label: {"size": int(np.sum(labels == label)), "top_families": "alpha, beta"}
                for label in unique
            },
        )

    return HierarchySpec(
        name="Classifier OOF", levels=[level("2 clusters", coarse), level("4 clusters", fine)]
    )


def test_validate_hierarchies_rejects_missing_colors():
    hierarchy = _make_hierarchy()
    bad_level = HierarchyLevel(
        name="bad",
        labels=hierarchy.levels[0].labels,
        colors={},
        metadata=hierarchy.levels[0].metadata,
    )
    with pytest.raises(ValueError, match="missing colors"):
        validate_hierarchies([HierarchySpec(name="bad", levels=[bad_level])], N_SAMPLES)


def test_plot_embedding_adds_discrete_hierarchy_controls():
    Z, contributions, feature_names, group_names = _make_inputs()
    layout = plot_embedding(
        Z,
        contributions,
        group_names=group_names,
        feature_names=feature_names,
        hierarchies=[_make_hierarchy()],
    )
    radios = list(layout.select({"type": RadioButtonGroup}))
    assert any("Hierarchy" in radio.labels for radio in radios)
    sliders = list(layout.select({"type": Slider}))
    hierarchy_slider = next(
        slider for slider in sliders if slider.title.startswith("Cluster depth")
    )
    assert hierarchy_slider.start == 0
    assert hierarchy_slider.end == 1
    selectors = list(layout.select({"type": Select}))
    assert any(selector.title == "Hierarchy source" for selector in selectors)
    divs = list(layout.select({"type": Div}))
    assert any("Classifier OOF" in div.text and "alpha, beta" in div.text for div in divs)
    assert layout.styles["width"] == "100%"
    assert layout.styles["height"] == "100vh"
    assert "max-width" not in layout.styles


def test_plot_embedding_adds_subset_legend_and_collapse_controls():
    Z, contributions, feature_names, group_names = _make_inputs()
    feature_names[0] = "1_ref_region_CDS"
    feature_names[1] = "2_ref_region_CDS"
    layout = plot_embedding(
        Z,
        contributions,
        group_names=group_names,
        feature_names=feature_names,
    )
    checkboxes = list(layout.select({"type": CheckboxGroup}))
    assert any(box.labels == ["Select none"] for box in checkboxes)
    assert any("Collapse numbered features" in box.labels[0] for box in checkboxes)
    divs = list(layout.select({"type": Div}))
    assert any("Color key" in div.text for div in divs)
