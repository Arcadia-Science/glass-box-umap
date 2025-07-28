# -*- coding: utf-8 -*-
"""
Refactored script for defining and demonstrating various PyTorch neural network modules.
"""

# --- Core Libraries ---
import torch
import torch.nn as nn
import math

# --- Helper & Custom Layer Definitions ---

class LayerNormDetached(nn.Module):
    """
    A LayerNorm implementation where the variance calculation is detached from the
    computation graph during evaluation, potentially stabilizing training.
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for LayerNormDetached."""
        mean = x.mean(dim=-1, keepdim=True)
        # Detach variance calculation during evaluation
        if not self.training:
            var = x.clone().detach().var(dim=-1, keepdim=True, unbiased=False)
        else:
            var = x.var(dim=-1, keepdim=True, unbiased=False)

        norm_x = (x - mean) / torch.sqrt(var)
        return self.scale * norm_x

class BilinearDetached(nn.Module):
    """
    A custom Bilinear layer where weights and inputs can be detached during evaluation.
    Computes: y = x1^T W x2 + b
    """
    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in1_features, in2_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform initialization."""
        bound = 1 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """Forward pass for BilinearDetached."""
        # Detach inputs and weights during evaluation
        if not self.training:
            output = torch.bilinear(input1.clone().detach(), input2, self.weight.clone().detach(), self.bias)
        else:
            output = torch.bilinear(input1, input2, self.weight, self.bias)
        return output

class Bilinear(nn.Module):
    """
    A composite Bilinear layer that combines a bilinear transformation
    with a parallel MLP path, followed by LayerNorm.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.bilinear = BilinearDetached(input_dim, input_dim, output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim, bias=False),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim, bias=False),
            nn.LeakyReLU(),
            nn.Linear(input_dim, output_dim, bias=False),
        )
        self.ln = LayerNormDetached(output_dim)
        self.act = nn.LeakyReLU()

        # Custom weight scaling
        scale = 1e-2 / input_dim
        self.bilinear.weight.data *= scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combines bilinear and MLP outputs."""
        bilinear_out = self.bilinear(x, x)
        mlp_out = self.mlp(x)
        return self.ln(self.act(bilinear_out) + mlp_out)


# --- Main Network Architecture Definitions ---

class deepReLUNet(nn.Module):
    """A deep network using PReLU activation."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False), nn.PReLU(),
            nn.Linear(hidden_size, hidden_size, bias=False), nn.PReLU(),
            nn.Linear(hidden_size, hidden_size, bias=False), nn.PReLU(),
            nn.Linear(hidden_size, hidden_size, bias=False), nn.PReLU(),
            nn.Linear(hidden_size, hidden_size, bias=False), nn.PReLU(),
            nn.Linear(hidden_size, output_size, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the PReLU network."""
        return self.model(x)

class deepSiLUNet(nn.Module):
    """A deep network using SiLU activation with a detached gate for evaluation."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden0 = nn.Linear(input_size, hidden_size, bias=False)
        self.hidden1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden3 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden4 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, output_size, bias=False)

    def _silu_with_detached_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Applies SiLU, detaching the sigmoid gate during evaluation."""
        if not self.training:
            return x * torch.sigmoid(x).clone().detach()
        return torch.nn.functional.silu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SiLU network."""
        h = self._silu_with_detached_gate(self.hidden0(x))
        h = self._silu_with_detached_gate(self.hidden1(h))
        h = self._silu_with_detached_gate(self.hidden2(h))
        h = self._silu_with_detached_gate(self.hidden3(h))
        h = self._silu_with_detached_gate(self.hidden4(h))
        return self.output(h)

class deepBilinearNet(nn.Module):
    """A deep network composed of sequential Bilinear layers."""
    def __init__(self, input_size: int = 50, n_components: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            Bilinear(input_size, 32),
            Bilinear(32, 16),
            Bilinear(16, 16),
            nn.Linear(16, n_components, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the bilinear encoder."""
        return self.encoder(x)

# --- Factory and Demonstration Functions ---

def create_model(model_class, device, **kwargs):
    """
    Factory function to create a model and move it to the specified device.

    Args:
        model_class: The model class to instantiate (e.g., deepReLUNet).
        device: The torch device ('cpu' or 'cuda') to move the model to.
        **kwargs: Arguments to pass to the model's constructor.

    Returns:
        An instance of the model on the specified device.
    """
    model = model_class(**kwargs)
    model.to(device)
    print(f"✅ Created {model_class.__name__} on {device}.")
    return model

# def run_demonstration():
#     """
#     Instantiates each model, prints its architecture, and runs a test forward pass.
#     """
#     print("🚀 Starting neural network demonstration...")
#     # --- Parameters ---
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     batch_size = 10
#     input_size = 50
#     hidden_size = 64
#     output_size = 2
    
#     # Create a dummy input tensor
#     dummy_input = torch.randn(batch_size, input_size).to(device)
#     print(f"Using dummy input of shape: {dummy_input.shape}\n")

#     # --- Instantiate and Test Models ---
    
#     # 1. Deep PReLU Network
#     relu_net = create_model(deepReLUNet, device, input_size=input_size, hidden_size=hidden_size, output_size=output_size)
#     print(relu_net)
#     with torch.no_grad():
#         output = relu_net(dummy_input)
#     print(f"Output shape: {output.shape}\n" + "-"*50)

#     # 2. Deep SiLU Network
#     silu_net = create_model(deepSiLUNet, device, input_size=input_size, hidden_size=hidden_size, output_size=output_size)
#     print(silu_net)
#     with torch.no_grad():
#         output = silu_net(dummy_input)
#     print(f"Output shape: {output.shape}\n" + "-"*50)

#     # 3. Deep Bilinear Network
#     bf_net = create_model(bfBilinearNet, device, input_size=input_size, n_components=output_size)
#     print(bf_net)
#     with torch.no_grad():
#         output = bf_net(dummy_input)
#     print(f"Output shape: {output.shape}\n" + "-"*50)
    
#     print("🎉 Demonstration complete.")

