# GitHub Actions

This document describes the GitHub Actions workflows that are used to build and publish the notebook publications.

## Overview

The publishing workflow described in the [Publishing Guide](PUBLISHING_GUIDE.md) is implemented using two GitHub Actions workflows:

1. **A build workflow** (`.github/workflows/build.yml`)

    This workflow creates the publication artifacts by aggregating all tagged versions of the `index.ipynb` notebook.

2. **A publishing workflow** (`.github/workflows/publish.yml`)
    
    This workflow renders the Quarto site from the publication artifacts and publishes the site to GitHub Pages.

## The version aggregation problem

The challenge addressed by the `build.yml` workflow is **version aggregation**. This refers to the fact that the public Quarto site (that is, the public publication) needs to show multiple historical versions of the notebook `index.ipynb`, one for each tagged release in the GitHub repository. Because GitHub tags correspond to commits, these versions exist in different commits in the repository. This means we can't render the final publication from a single commit or branch. Instead, we need to create a special branch that contains all of these versions together.

## The build workflow (`build.yml`)

This workflow solves the version aggregation problem by doing two things:

1. It runs the [`_build.py`](../_build.py) script. This script:
   - Extracts the `index.ipynb` notebook from each tag in the repository.
   - Copies each version to a new uniquely named file (e.g., `index_v1.ipynb`, `index_v2.ipynb`).
   - Collects all execution results from each tag's `_freeze` directory.
   - Updates `_quarto.yml` to include a version selector menu for all tagged versions.
   - Renames the most recent tagged version of the notebook to `index.ipynb`.

2. It opens a PR to merge the notebook versions and execution results into a special `publish` branch. The `publish` branch is used only to store the publication artifacts, not for version control, so it is reset to match `main` before the PR is opened.

This workflow is triggered whenever a new tag is pushed to the repository (e.g., `v1`, `v2`). It can also be manually triggered.

## The publishing workflow (`publish.yml`)

This workflow is more straightforward. It takes the aggregated content from the `publish` branch and renders it with Quarto. It then publishes the rendered content to GitHub Pages via the `gh-pages` branch. After this, the Quarto site will be live on the internet at `arcadia-science.github.io/<your-repo-name>`.

This workflow runs whenever a PR created by the build workflow (described above) is merged into the `publish` branch.

## Overall development and publication process

The full development and publication process is described briefly below to help contextualize the GitHub Actions workflows in terms of the overall publication process.

1. **Development**: Contributors merge changes to `main` via PRs as usual.

2. **Versioning**: When the author is ready to publish their notebook (either for the first time or as a revision), they create a new tag (`v1` for the first publication, `v2` for the first revision, etc.). See the [Publishing Guide](PUBLISHING_GUIDE.md) for more details.

3. **Building the publication**: When the author pushes a new tag, the build workflow is triggered. It aggregates all tagged notebook versions and opens a PR to merge them into a special `publish` branch. The author can then review this PR and merge it into `publish` when they are ready to publish the publication.

4. **Publishing the publication**: When the author merges the build PR into `publish`, the `publish` workflow is triggered. It renders all of the versions and deploys the Quarto site to GitHub Pages.

