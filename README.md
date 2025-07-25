# Notebook pub template

This repo is a template for notebook publications. The publication rendered and hosted by this repository serves as a demo that can be viewed at [this URL](https://arcadia-science.github.io/notebook-pub-template/).

## How to use this template

1. Create a repo from this template

    In the top-right of this GitHub repo, select the green button that says "*Use this template*".

    **IMPORTANT**: When creating your repo from this template, you need to **check the box** that says, "*Include all branches*". This is because the hosted pub is managed on a separate branch.

1. Config edits

    * Replace the variables in `_variables.yml`
        - For `google_analytics`, provide the publishing team your repo name and they'll provide you a Google Analytics tracking ID.
    * Replace the variables in `authors.yml`.
        - This can always be updated, but for now at least add yourself.

1. Install Quarto

    The publication is rendered with [Quarto](https://quarto.org/). If you don’t have it installed (check with quarto --version), you can install it [here](https://quarto.org/docs/get-started/).

1. Setup the code environment

    This repository uses conda to manage the computational and build environment. If you don’t have it installed (check with `conda --version`), you can find operating system-specific instructions for installing miniconda [here](https://docs.anaconda.com/miniconda/).

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

    And finally, install the [pre-commit](https://pre-commit.com/) hooks:

    ```bash
    pre-commit install
    ```

    Test your installation with `make preview`. Your pub will open up in your browser.

    Afterwards, create a branch to work on (don't commit to `main` directly).

1. Register your publication with the Pub Team

    If you intend to publish your analysis, fill out the "*Kick off a new pub*" form on the AirTable [Publishing toolkit](https://www.notion.so/arcadiascience/Publishing-2-0-f0c51bf29d1d4356a86e6cf8a72ae88b?pvs=4#e1de83e8dd2a4081904064347779ed25).

1. Create your pub

    Edit `index.ipynb` to create your pub. As you work, render it in a live preview with `make preview`.


## How to publish

1. Enable read/write permissions for GitHub Actions

    In your repo, go to *Settings* -> *Actions* -> *General* -> *Workflow permissions*, and check the box, "*Read and write permissions*"

1. Populate the `README_TEMPLATE.md`

    Populate `README_TEMPLATE.md` and then rename it to `README.md`.

    **NOTE**: The content you're reading now is the current `README.md`, which is to be replaced with `README_TEMPLATE.md`.

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

    In order for this pub to be open and reproducible, make the [repo public](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility).

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

## Publishing revisions

See [CONTRIBUTING.qmd](CONTRIBUTING.qmd).
