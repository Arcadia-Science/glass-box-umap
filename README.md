# Glass Box UMAP

This repo contains a Python package called `glass_box_umap`.

This package augments UMAP by computing exact feature contributions to the UMAP embedding.

The publication for Glass Box UMAP can be found [here](https://arcadia-science.github.io/glass-box-umap-notebook-pub/).

## Installation

```bash
# pip
pip install glass-box-umap

# uv
uv pip install glass-box-umap
```

Find complete installation instructions at `docs/install.md`.

## Development

### Environment setup

We use [uv](https://docs.astral.sh/uv/) for dependency management and build tooling. First, install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project with development dependencies:

```bash
uv sync --group dev
```

To also install documentation dependencies:

```bash
uv sync --group dev --group docs
```

This creates a virtual environment in `.venv` and installs all dependencies. The package itself is automatically installed in editable mode (equivalent to `pip install -e .`).

The easiest way to run code is to simply prefix commands with `uv run` (e.g., `uv run python main.py`). This executes the command inside the virtual environment automatically, so you don't need to activate it first.

Alternatively, if you prefer a traditional workflow, or are running from a different directory, you can manually source the activation script:

```bash
source .venv/bin/activate
```

### Formatting and linting

To format the code, use the following command:

```bash
make format
```

To run the lint checks and type checking, use the following command:

```bash
make lint
```

### Pre-commit hooks

We use pre-commit to run formatting and lint checks before each commit. To install the pre-commit hooks, use the following command:

```bash
pre-commit install
```

To run the pre-commit checks manually, use the following command:

```bash
make pre-commit
```

### Testing

We use `pytest` for testing. The tests are found in the `glass_box_umap/tests/` subpackage. To run the tests, use the following command:

```bash
make test
```

### Managing dependencies

We use uv to manage dependencies. To add a new dependency:

```bash
uv add some-package
```

To add a new development dependency:

```bash
uv add --group dev some-dev-package
```

To update a dependency:

```bash
uv lock --upgrade-package some-package
```

Whenever you add or update a dependency, uv will automatically update both `pyproject.toml` and `uv.lock`. Make sure to commit changes to these files.

### Documentation

We use Sphinx for documentation with the [`furo` theme](https://github.com/pradyunsg/furo). We also use some Sphinx extensions (described below) to make the process of writing documentation easier.

To build the docs, first install `pandoc`. On macOS, this can be done using `brew`:

```
brew install pandoc
```

Then, build the docs using the following command:

```bash
make docs
```

Note: the `pandoc` dependency is only required by the `nbsphinx` extension. If this extension is removed, there is no need to install `pandoc`.

#### sphinx-autoapi

This extension generates API docs automatically from the docstrings in the source code. To do so, it requires that docstrings adhere to the Google or Numpy style. This style is described in the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

#### napolean

Rather than writing our docstrings in RST, we use this extension to convert Google and NumPy-style docstrings to RST at build time.

#### myst-parser

RST is complicated to write. This extension lets us write our docs in Markdown and then converts them to RST at build time.

#### nbsphinx

It is often convenient to write examples as Jupyter notebooks. This extension executes Jupyter notebooks and renders the results in the docs at build time. It requires `pandoc`, which can be installed using `brew install pandoc`.

#### Removing unused Sphinx extensions

To remove an unused extension, delete the corresponding line from the `extensions` list in `docs/conf.py` and delete the extension from the `docs` dependency group in `pyproject.toml`.

## Publishing the package on PyPI

Publishing the package on PyPI requires that you have API tokens for the test and production PyPI servers. You can find these tokens in your PyPI account settings. Create a `.env` file by coping `.env.copy` and add your tokens to this file.

We use semantic versioning of the form `MAJOR.MINOR.PATCH`. See [semver.org](https://semver.org/) for more information. When you're ready to release a new version:

1. Update the `version` field in `pyproject.toml` to the new version number
2. Commit the change: `git commit -am "Bump version to X.Y.Z"`
3. Create a git tag matching the version:

```bash
RELEASE_VERSION=0.1.0
git tag -a v${RELEASE_VERSION} -m "Release version ${RELEASE_VERSION}"
git push origin v${RELEASE_VERSION}
```

__Before creating the tag, make sure that your local git repository is on `main`, is up-to-date, and does not contain uncommitted changes!__

If you need to delete a tag you've created:

```bash
git tag -d v${RELEASE_VERSION}
```

If you already pushed the deleted tag to GitHub, also delete it from the remote:

```bash
git push origin :refs/tags/v${RELEASE_VERSION}
```

Once you've created the correct new tag, build the package:
```bash
make build
```

You should see an output that looks like this:
```
Building glass-box-umap (0.1.0)
  - Building sdist
  - Built glass-box-umap-0.1.0.tar.gz
  - Building wheel
```

The build artifacts are written to the `dist/` directory.

__Make sure that the version number in the output from `make build` matches the one in `pyproject.toml` and the git tag!__

If it does not, double-check that you updated the `version` field in `pyproject.toml` before creating the tag.

Next, check that you can publish the package to the PyPI test server:

```bash
make build-and-test-publish
```

The `build-and-test-publish` command calls `uv build` to build the package and then `uv publish` to upload the build artifacts to the test server.

Check that you can install the new version of the package from the test server:

```bash
pip install --index-url https://pypi.org/simple/ --extra-index-url https://test.pypi.org/simple/ glass-box-umap==${RELEASE_VERSION}
```

(For accurate installation testing: dependencies are installed from main PyPI (to avoid unreliable TestPyPI mirrors), while your package (not yet on main PyPI) is installed from TestPyPI.)

If everything looks good, build and publish the package to the prod PyPI server:

```bash
make build-and-publish
```

Finally, check that you can install the new version of the package from the prod PyPI server:

```bash
pip install glass-box-umap==${RELEASE_VERSION}
```
