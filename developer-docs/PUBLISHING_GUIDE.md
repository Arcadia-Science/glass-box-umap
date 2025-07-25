**FIXME This needs to be updated once the git tag versioning system lands**

# Publishing Guide

This document provides a step-by-step explanation for publishing a notebook publication.

Instructions are provided for:

* Initial publication
* Revision publications

## Steps for initial publication

1. Enable read/write permissions for GitHub Actions

    In your repo, go to *Settings* -> *Actions* -> *General* -> *Workflow permissions*, and check the box, "*Read and write permissions*"

1. Populate the publication information in the `README.md`

1. Remove placeholder package

    If you did not populate `src/analysis` with your own content, remove it (`rm -rf src/analysis`).

1. Remove reference to the syntax demo from `_quarto.yml`

    Feel free to keep `demo.ipynb` in your repo, but remove the following lines in `_quarto.yml`:

    ```
    - text: 'Demo'
      href: demo.ipynb
    ```

    This will remove *Demo* from the navigation bar.

1. Make the repository public

    In order for this pub to be open and reproducible, make the [repo public](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility). Be sure it meets [our standards](https://github.com/Arcadia-Science/arcadia-software-handbook/blob/main/guides-and-standards/standards--public-repos.md) for public-facing repos.

1. Enable comments

    Comments are handled with [Giscus](https://giscus.app/), which Quarto has an integration for. Once enabled, a widget is placed at the bottom of the publication that provides an interface to read, write, react, and respond to [GitHub Discussions](https://docs.github.com/en/discussions) comments. Comments made through the interface are automatically added as comments to a GitHub Discussions thread of your repository.

    First, [enable GitHub Discussions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/enabling-or-disabling-github-discussions-for-a-repository) for your repo.

    Second, [install the Giscus App](https://github.com/apps/giscus) for your repository. Click *Configure*, select *Arcadia-Science*, then select your repository from the dropdown. Click *Update access*.

    **IMPORTANT**: Do not deselect any of the other Arcadia-Science repositories that already have the Giscus app installed, *e.g.* `Arcadia-Science/notebook-pub-template`.

    Now, edit the comments section in `_quarto.yml` with your repo name:

    ```yaml
    comments:
      giscus:
        repo: Arcadia-Science/notebook-pub-template
        input-position: top
    ```

    You may have to wait a few minutes for `make preview` to properly render the Giscus widget.

1. Get approval from the Pub Team

    Like all other pubs, follow the [AirTable toolkit guide](https://airtable.com/appN7KQ55bT6HHfog/pagm69ti1kZK1GhBx) through to the final step, "*Submit your pub for release*".

    * Once all contributor roles have been assigned, populate `_authors.yml` accordingly.
    * Once all required authors sign the *AI methods form*, paste the following lines at the end of your section with the heading `## Abstract`:

        ```
        ----

        :::{.callout-note title="AI usage disclosure" collapse="true"}
        This is a placeholder for the AI usage disclosure. Once all authors sign the AI code form on AirTable, SlackBot will message you an AI disclosure that you should place here.
        :::
        ```

1. Final run-through

    Begin with a clean branch (no uncommitted changes). Then run the notebook from the command line:

    ```bash
    make execute
    ```

    This command will update `index.ipynb` with the latest execution results. Importantly, it may generate runtime artifacts in the `_freeze/` directory.

    Then run `make preview` to see how the publication is rendering. Verify that your changes appear how you intend them to appear. If not, make the necessary changes and re-run `make execute`.

    Once happy, commit `index.ipynb` and all files in the `_freeze/` directory.

    Now, create a pull request to merge your branch into `main`. Once your PR is approved, merge into `main`.

1. Host the publication

    Publishing is automated through a GitHub Action that triggers when a pull request is merged into the `publish` branch. Thus, all that's required for your publication to go live is to merge into `publish`. By convention, we only merge changes from the `main` branch into `publish`. This ensures collaborators can merge their completed work into `main`, where others can see and build upon it, while keeping those changes private until they are deliberately hosted by merging `main` into `publish`.

    When all your changes have been merged into `main` and you're ready for your publication to go live, open a pull request to merge `main` into `publish`. Once approved and merged, your publication will be live within minutes.

## Steps for publishing a revision

For information on publishing revisions to an existing publication, see [CONTRIBUTING.qmd](../CONTRIBUTING.qmd).
