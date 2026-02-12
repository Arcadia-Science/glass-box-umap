# Quickstart

Glass Box UMAP works like standard parametric UMAP but adds the ability to compute exact feature contributions to the embedding.

## Basic Usage

```python
import torch
from glass_box_umap import GlassBoxUMAP

# Your data as a torch tensor
X = torch.randn(1000, 50)

# Fit the model
model = GlassBoxUMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    epochs=50,
)
embedding = model.fit_transform(X)
```

## Computing Feature Contributions

The main feature of Glass Box UMAP is computing exact attributions that explain how each input feature contributes to the embedding coordinates:

```python
# Compute feature contributions
contributions, jacobians = model.compute_attributions(X)

# contributions shape: (n_samples, n_components, n_features)
# contributions[i, j, k] = contribution of feature k to embedding dimension j for sample i
```

Each sample's embedding can be reconstructed exactly by summing its feature contributions:

```python
import numpy as np

# The contributions sum to the embedding (up to machine precision)
reconstructed = contributions.sum(axis=2)
np.allclose(reconstructed, embedding, atol=1e-5)  # True
```

## Interpreting Contributions

Feature contributions tell you which features are responsible for a point's position in the embedding:

```python
import matplotlib.pyplot as plt

# For a single sample, see which features drive each embedding dimension
sample_idx = 0
feature_names = [f"Feature {i}" for i in range(X.shape[1])]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for dim in range(2):
    ax = axes[dim]
    contrib = contributions[sample_idx, dim, :]
    ax.barh(feature_names[:10], contrib[:10])
    ax.set_xlabel("Contribution")
    ax.set_title(f"UMAP Dimension {dim + 1}")
plt.tight_layout()
```

## Using PCA Preprocessing

For high-dimensional data, you can apply PCA before the UMAP encoder. Contributions are automatically projected back to the original feature space:

```python
model = GlassBoxUMAP(
    n_neighbors=15,
    min_dist=0.1,
    pca_components=100,  # Reduce to 100 PCA components first
    epochs=50,
)
embedding = model.fit_transform(X)

# Contributions are still in original feature space
contributions, _ = model.compute_attributions(X)
# contributions.shape[2] == X.shape[1], not 100
```

## Saving and Loading Models

```python
from pathlib import Path

# Save the fitted model
model.save(Path("glass_box_umap_model.pt"))

# Load it back
loaded_model = GlassBoxUMAP.load(Path("glass_box_umap_model.pt"))
```
