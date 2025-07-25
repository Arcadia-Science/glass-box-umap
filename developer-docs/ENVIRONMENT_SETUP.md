# Environment Setup Guide

This document explains how to set up the development environment for working with and reproducing notebook publications.

## Setup the code environment

This repository uses conda to manage the computational and build environment. If you don't have it installed (check with `conda --version`), you can find operating system-specific instructions for installing miniconda [here](https://docs.anaconda.com/miniconda/).

When you're ready, run the following commands to create and activate the environment. Replace `[REPO-NAME]` with your repository name.

```bash
conda env create -n [REPO-NAME] --file env.yml
conda activate [REPO-NAME]
```

(As you introduce dependencies to your publication, or if you already have your full set of dependencies, add them to `env.yml` with the version pinned.)

Now, install any internal packages in the repository:

```bash
pip install -e .
```

And finally, install the [pre-commit](https://pre-commit.com/) hooks. This is optional but recommended:

```bash
pre-commit install
```

Test your installation with `make preview`. Your pub will open up in your browser.

Afterwards, create a branch to work on (don't commit to `main` directly).

## Pub Development

When working with Jupyter notebooks:

1. Edit `index.ipynb` as your main publication file
2. Use `make preview` to see changes in real-time
3. Run `make execute` before committing to update `_freeze/` with execution results

## Other Commands

The Makefile provides several useful commands for development:

- `make lint`: Run linting checks using ruff
- `make format`: Automatically format code using ruff
- `make pre-commit`: Run all pre-commit hooks
- `make test`: Run pytest tests, if they exist
- `make bump-version`: Update version number
