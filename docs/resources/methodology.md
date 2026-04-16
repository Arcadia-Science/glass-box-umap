# Methodology

This page explains how Glass Box UMAP computes exact feature contributions to UMAP embeddings.

## The Locally Linear Property

Glass Box UMAP uses a neural network architecture that is *locally linear*: for any input $x$, the network's output can be expressed as a linear function of that input. Specifically, if $f$ is the network function:

$$f(x) = J(x) \cdot x$$

where $J(x)$ is the Jacobian matrix of $f$ evaluated at $x$. This means the output is exactly the Jacobian times the input—not an approximation, but an equality.

## Why This Property Holds

The locally linear property emerges from two architectural choices:

1. **PReLU activations**: The PReLU function $\text{PReLU}(z) = \max(0, z) + \alpha \min(0, z)$ is piecewise linear. For any fixed input, PReLU acts as a linear scaling operation.

2. **Zero-bias linear layers**: All linear layers use `bias=False`. This ensures the network passes through the origin: $f(0) = 0$.

When you compose piecewise linear functions (PReLU) with linear functions (bias-free matrix multiplications), the result is itself piecewise linear. Combined with the zero-origin property, this means for any input $x$, there exists a matrix $A(x)$ such that $f(x) = A(x) \cdot x$. This matrix $A(x)$ is exactly the Jacobian $J(x)$.

## Computing Feature Contributions

Given the locally linear property, computing exact feature contributions is straightforward:

1. **Compute the Jacobian** $J(x)$ via automatic differentiation. The Jacobian has shape $(d_{out}, d_{in})$ where $d_{out}$ is the embedding dimension (typically 2) and $d_{in}$ is the input dimension.

2. **Multiply element-wise** with the input: $C_{ij} = J_{ij} \cdot x_j$

3. **The contributions sum exactly** to the embedding: $y_i = \sum_j C_{ij}$

The contribution $C_{ij}$ tells you exactly how much feature $j$ contributes to embedding dimension $i$ for this sample.

## Handling PCA Preprocessing

For high-dimensional data, Glass Box UMAP optionally applies PCA before the neural network encoder. The Jacobian computation accounts for this:

1. The input $x$ is centered and projected to PCA space: $x_{pca} = P \cdot (x - \mu)$
2. The encoder operates on $x_{pca}$: $y = f(x_{pca})$
3. The Jacobian w.r.t. PCA features is $J_{pca} = \partial y / \partial x_{pca}$
4. To get contributions in original feature space, project back: $J_{raw} = J_{pca} \cdot P$

This ensures contributions are always expressed in terms of the original input features, not PCA components.

## Validation: Machine-Precision Accuracy

The locally linear property can be verified empirically. For any sample, the sum of feature contributions should equal the embedding exactly. In practice, Glass Box UMAP achieves reconstruction errors on the order of $3 \times 10^{-14}$—machine precision for 64-bit floating point.

Glass Box UMAP's contributions are exact, not estimates.

## Further Reading

For the complete methodology and validation experiments, see the [Glass Box UMAP publication](https://arcadia-science.github.io/glass-box-umap-notebook-pub/).
