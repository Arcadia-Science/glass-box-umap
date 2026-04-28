from dataclasses import fields

import numpy as np
import torch
from sklearn.decomposition import PCA

from .lightning import UMAPLightningModule

FIELDS_WITH_CUSTOM_EQ = frozenset({"_model", "_pca", "_mean", "_device"})
FIELDS_IGNORED_FOR_EQ = frozenset({"quiet", "extra_callbacks", "num_workers", "checkpoint_dir"})


def state_dicts_equal(a: UMAPLightningModule, b: UMAPLightningModule) -> bool:
    sa = a.state_dict()
    sb = b.state_dict()
    if sa.keys() != sb.keys():
        return False
    for key, ta in sa.items():
        tb = sb[key]
        if ta.shape != tb.shape or ta.dtype != tb.dtype:
            return False
        if not torch.equal(ta.cpu(), tb.cpu()):
            return False
    return True


def models_equal(a: UMAPLightningModule | None, b: UMAPLightningModule | None) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if type(a) is not type(b):
        return False
    return state_dicts_equal(a, b)


def pcas_equal(a: PCA | None, b: PCA | None) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return np.array_equal(a.components_, b.components_) and np.array_equal(a.mean_, b.mean_)


def parametric_umap_equal(a, b) -> bool:  # type: ignore
    """Semantic equality between two ParametricUMAP-like dataclasses.

    "Semantic" in the sense that the comparison concerns the trained artifact and how it
    was produced — the recipe (provenance hyperparameters, architecture) and the result
    (learned weights, PCA fit, centering vector), not the runtime envelope around it
    (device, logging verbosity, DataLoader parallelism, etc.). Two instances that
    describe and contain the same trained model compare equal even if they differ in
    incidental runtime state. Where the function *does* compare values, the comparison
    is bitwise-exact (no FP tolerance).
    """
    if a.__class__ is not b.__class__:
        return False
    for f in fields(a):
        if f.name in FIELDS_WITH_CUSTOM_EQ or f.name in FIELDS_IGNORED_FOR_EQ:
            continue
        if getattr(a, f.name) != getattr(b, f.name):
            return False
    if not models_equal(a._model, b._model):
        return False
    if not pcas_equal(a._pca, b._pca):
        return False
    if not np.array_equal(a._mean, b._mean):
        return False
    return True
