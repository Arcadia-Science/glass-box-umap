:::{admonition} Glass Box UMAP isn't live!
:class: Danger

These installation instructions will fail because Glass Box UMAP is not yet on PyPI!
:::

---

# Installation

Glass Box UMAP is available on the [Python Package Index (PyPI)](https://pypi.org/project/glass-box-umap/).

:::{admonition} PyTorch Dependency
:class: warning

Glass Box UMAP requires [PyTorch](https://pytorch.org/).

* **Existing Install:** We respect and utilize any compatible version of PyTorch already found in your environment.
* **New Install:** If PyTorch is missing, a compatible version will be automatically installed.
:::

## With uv

```{eval-rst}
.. tabs::

   .. tab:: **Linux**

      .. code-block:: bash

         uv pip install glass-box-umap --torch-backend=auto

      *(If PyTorch is not found, installs version optimized for your hardware)*

   .. tab:: **MacOS**

      .. code-block:: bash

         uv pip install glass-box-umap --torch-backend=auto

      *(If PyTorch is not found, installs version optimized for your hardware)*

   .. tab:: **Windows**

      .. code-block:: bash

         uv pip install glass-box-umap --torch-backend=auto

      *(If PyTorch is not found, installs version optimized for your hardware)*
```

## With pip


```{eval-rst}
.. tabs::

   .. tab:: **Linux**

      .. code-block:: bash

         pip install glass-box-umap

      *(If PyTorch is not found, installs CUDA-enabled version)*

   .. tab:: **MacOS**

      .. code-block:: bash

         pip install glass-box-umap

      *(If PyTorch is not found, installs CPU/MPS compatible version)*

   .. tab:: **Windows**

      .. code-block:: bash

         pip install glass-box-umap

      *(If PyTorch is not found, installs CPU-only version)*

      .. warning:: **Using an NVIDIA GPU on Windows?**

         To use a CUDA-enabled version of PyTorch, install separately before installing Glass Box UMAP:

         .. code-block:: bash

            # Example for CUDA 12.4
            pip install torch --index-url https://download.pytorch.org/whl/cu124

            # Then install glass-box-umap
            pip install glass-box-umap
```
