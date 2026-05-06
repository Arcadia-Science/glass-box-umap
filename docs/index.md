# Glass Box UMAP

Glass Box UMAP augments UMAP by computing exact feature contributions to the UMAP embedding.

## Overview

Standard UMAP produces embeddings but offers no insight into why points land where they do. Glass Box UMAP solves this by using a specially designed neural network that enables exact computation of feature contributions—no approximations needed.

The key insight is that certain neural network architectures are *locally linear*: for any input, the network's output can be expressed exactly as a matrix multiplication of that input. Glass Box UMAP exploits this property by using PReLU activations and zero-bias linear layers, allowing us to compute the Jacobian of the embedding with respect to input features. Multiplying this Jacobian by the input gives exact feature contributions that sum precisely to the embedding coordinates.

Glass Box UMAP's feature contributions are mathematically exact, validated to near machine precision. This makes it possible to understand exactly which features drive the structure in your embeddings.

For a detailed explanation of the methodology, see the [Methodology](resources/methodology.md) page. For the full publication, visit the [Glass Box UMAP publication](https://arcadia-science.github.io/glass-box-umap-notebook-pub/).

```{eval-rst}
.. toctree::
   :hidden:
   :caption: Home

   self
   GitHub <https://github.com/Arcadia-Science/glass-box-umap>

.. toctree::
   :hidden:
   :caption: User Guide

   user_guide/install
   user_guide/basic_usage
   user_guide/saving_and_loading
   user_guide/monitoring_training
   user_guide/embedding_comparison
   user_guide/embedding_refinement

.. toctree::
   :hidden:
   :caption: Contents

   examples/index
   resources/index

.. toctree::
   :hidden:
   :caption: API Reference

   autoapi/glass_box_umap/index

.. toctree::
   :hidden:
   :caption: Meta

   meta/citation
   meta/contributing
   meta/license
```
