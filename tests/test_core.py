from glass_box_umap.components import LayerNormDetached
from glass_box_umap.core import DeepPReLUNet
from torch import nn


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
        nn.PReLU(),
        LayerNormDetached(hidden_size),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.PReLU(),
        LayerNormDetached(hidden_size),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.PReLU(),
        LayerNormDetached(hidden_size),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.PReLU(),
        LayerNormDetached(hidden_size),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.PReLU(),
        nn.Linear(hidden_size, n_components, bias=False),
    )

    # Direct comparison doesn't work, so we use `repr` as a shortcut to avoid
    # attribute-by-attribute comparison.
    assert repr(net.model) == repr(expected)
