# Contributing

## Environment setup

We use [uv](https://docs.astral.sh/uv/) for dependency management and build tooling. First, verify that uv is installed:

```bash
uv --version
```

If it's not, install it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone the repository and install the project with all development dependencies:

```bash
git clone https://github.com/Arcadia-Science/glass-box-umap.git
cd glass-box-umap
uv sync --extra plotting --group dev --group docs
```

:::{important}
It's essential that you include the `--extra` and `--group` parameters.
:::

This creates a virtual environment in `.venv` and installs all dependencies. The package itself is automatically installed in editable mode.

The easiest way to run code is to prefix commands with `uv run` (e.g., `uv run <YOUR_COMMAND>`). This executes the command inside the virtual environment automatically.

Alternatively, you can manually activate the virtual environment:

```bash
source .venv/bin/activate
```

## Formatting and linting

To format the code:

```bash
make format
```

To run lint checks and type checking:

```bash
make lint
```

## Pre-commit hooks

We use pre-commit to run formatting and lint checks before each commit. To install the hooks:

```bash
pre-commit install
```

To run the pre-commit checks manually:

```bash
make pre-commit
```

## Testing

We use `pytest` for testing. Tests are in the `glass_box_umap/tests/` subpackage. To run the tests:

```bash
make test
```

## Managing dependencies

To add a new dependency:

```bash
uv add some-package
```

To add a development dependency:

```bash
uv add --group dev some-dev-package
```

To update a dependency:

```bash
uv lock --upgrade-package some-package
```

Whenever you add or update a dependency, uv will automatically update both `pyproject.toml` and `uv.lock`. Commit changes to both files.

## Building documentation

We use Sphinx with the [furo](https://github.com/pradyunsg/furo) theme. First, install `pandoc` (required by nbsphinx).

```bash
# macOS
brew install pandoc

# Windows/Linux ???
```

Then build the docs:

```bash
make docs
```

## Publishing to PyPI

Publishing requires API tokens for the test and production PyPI servers. Create a `.env` file by copying `.env.copy` and add your tokens.

We use semantic versioning (`MAJOR.MINOR.PATCH`). See [semver.org](https://semver.org/) for details.

### Release process

1. Update the `version` field in `pyproject.toml`
2. Commit the change: `git commit -am "Bump version to X.Y.Z"`
3. Create a git tag:

```bash
RELEASE_VERSION=0.1.0
git tag -a v${RELEASE_VERSION} -m "Release version ${RELEASE_VERSION}"
git push origin v${RELEASE_VERSION}
```

Make sure your local git repository is on `main`, up-to-date, and has no uncommitted changes before creating the tag.

4. Build the package:

```bash
make build
```

Verify the version number in the output matches `pyproject.toml` and the git tag.

5. Test publish to PyPI test server:

```bash
make build-and-test-publish
```

6. Verify installation from test server:

```bash
pip install --index-url https://pypi.org/simple/ --extra-index-url https://test.pypi.org/simple/ glass-box-umap==${RELEASE_VERSION}
```

7. Publish to production PyPI:

```bash
make build-and-publish
```

8. Verify installation from production in a fresh, isolated environment:

```bash
pip install glass-box-umap==${RELEASE_VERSION}
```

### Deleting a tag

If you need to delete a tag:

```bash
git tag -d v${RELEASE_VERSION}
```

If already pushed to GitHub:

```bash
git push origin :refs/tags/v${RELEASE_VERSION}
```
