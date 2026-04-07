import torch
import torch.nn as nn
from glass_box_umap.components import VmapPreLU
from torch.func import functional_call, jacrev, vmap


def test_train_mode_matches_prelu():
    x = torch.randn(10, 5)
    slope = 0.3

    vmap_prelu = VmapPreLU(init=slope)
    prelu = nn.PReLU(init=slope)

    vmap_prelu.train()
    prelu.train()

    assert torch.allclose(vmap_prelu(x), prelu(x))


def test_eval_mode_matches_prelu():
    x = torch.randn(10, 5)
    slope = 0.3

    vmap_prelu = VmapPreLU(init=slope)
    prelu = nn.PReLU(init=slope)

    vmap_prelu.eval()
    prelu.eval()

    assert torch.allclose(vmap_prelu(x), prelu(x))


def test_gradient_flows_in_train_mode():
    x = torch.randn(10, 5)
    vmap_prelu = VmapPreLU()
    vmap_prelu.train()

    out = vmap_prelu(x).sum()
    out.backward()

    assert vmap_prelu.weight.grad is not None


def test_vmap_jacrev_compatible():
    model = nn.Sequential(nn.Linear(5, 3, bias=False), VmapPreLU(), nn.Linear(3, 2, bias=False))
    model.eval()

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def func_single(x_single: torch.Tensor) -> torch.Tensor:
        return functional_call(model, {**params, **buffers}, (x_single.unsqueeze(0),)).squeeze(0)

    jac_fn = vmap(jacrev(func_single))
    x = torch.randn(4, 5)

    with torch.no_grad():
        J = jac_fn(x)

    assert J.shape == (4, 2, 5)


def test_cache_invalidated_on_train():
    vmap_prelu = VmapPreLU(init=0.3)
    vmap_prelu.eval()

    x = torch.randn(5)
    vmap_prelu(x)
    assert vmap_prelu._cached_slope is not None

    vmap_prelu.train()
    assert vmap_prelu._cached_slope is None


def test_cache_refreshed_after_weight_change():
    x = torch.randn(10, 5)
    vmap_prelu = VmapPreLU(init=0.25)

    vmap_prelu.eval()
    vmap_prelu(x)
    old_slope = vmap_prelu._cached_slope

    vmap_prelu.train()
    with torch.no_grad():
        vmap_prelu.weight.fill_(0.5)
    vmap_prelu.eval()
    vmap_prelu(x)

    assert vmap_prelu._cached_slope == 0.5
    assert vmap_prelu._cached_slope != old_slope
