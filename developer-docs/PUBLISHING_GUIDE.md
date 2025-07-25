# Publishing Guide

This document provides a step-by-step explanation for publishing a notebook publication.

Instructions are provided for:

* Publishing the initial publication
* Publishing a revision

## Steps for initial publication

1. **Enable read/write permissions for GitHub Actions**

    In your repo, go to *Settings* -> *Actions* -> *General* -> *Workflow permissions*, and check the box, "*Read and write permissions*".

1. **Populate the publication information in the `README.md`**

1. **[If necessary] Remove the placeholder package**

    If you did not populate `src/analysis` with your own content, remove it using `rm -rf src/analysis`.

1. **Remove references to the demo notebook from `_quarto.yml`**

    Remove the following lines from `_quarto.yml`:

    ```
    - text: 'Demo'
      href: demo.ipynb
    ```

    This will remove the *"Demo"* link from the navigation bar.

1. **Make the repository public**

    In order for this pub to be open and reproducible, make the [repo public](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility). Be sure it meets [our standards](https://github.com/Arcadia-Science/arcadia-software-handbook/blob/main/guides-and-standards/standards--public-repos.md) for public-facing repos.

1. **Enable comments**

    Comments are handled with [Giscus](https://giscus.app/), which Quarto has an integration for. Once enabled, a widget is placed at the bottom of the publication that provides an interface to read, write, react, and respond to [GitHub Discussions](https://docs.github.com/en/discussions) comments. Comments made through the interface are automatically added as comments to a GitHub Discussions thread of your repository.

    First, [enable GitHub Discussions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/enabling-or-disabling-github-discussions-for-a-repository) for your repo.

    Second, [install the Giscus App](https://github.com/apps/giscus) for your repository. Click *Configure*, select *Arcadia-Science*, then select your repository from the dropdown. Click *Update access*.

    **IMPORTANT**: Do not deselect any of the other Arcadia-Science repositories that already have the Giscus app installed, *e.g.* `Arcadia-Science/notebook-pub-template`.

    Now, edit the comments section in `_quarto.yml` with your repo name:

    ```yaml
    comments:
      giscus:
        repo: Arcadia-Science/<your-repo-name>
        input-position: top
    ```

    You may have to wait a few minutes for `make preview` to properly render the Giscus widget.

1. **Make sure the notebook outputs are up to date in `index.ipynb`**

    Begin with a clean branch (no uncommitted changes). Then run the notebook from the command line using the `make execute` command.

    This command will update `index.ipynb` with the latest execution results. Importantly, it may generate runtime artifacts in the `_freeze/` directory.

    Then run `make preview` to see how the publication is rendering. Verify that your changes appear how you intend them to appear. If not, make the necessary changes and re-run `make execute`.

    Once everything looks good, commit `index.ipynb` and all files in the `_freeze/` directory.

    Now, create a pull request to merge your branch into `main`. Once your PR is approved, merge into `main`.

1. **Get approval from the Pub Team**

    Like all other pubs, follow the [AirTable toolkit guide](https://airtable.com/appN7KQ55bT6HHfog/pagm69ti1kZK1GhBx) through to the final step, "*Submit your pub for release*".

    * Once all contributor roles have been assigned, populate `_authors.yml` accordingly.
    * Once all required authors sign the *AI methods form*, paste the following lines at the end of your section with the heading `## Abstract`:

        ```
        ----

        :::{.callout-note title="AI usage disclosure" collapse="true"}
        This is a placeholder for the AI usage disclosure. Once all authors sign the AI code form on AirTable, SlackBot will message you an AI disclosure that you should place here.
        :::
        ```

1. **Create a tagged release of your repo**

    Follow the instructions on Notion [here](https://www.notion.so/arcadiascience/How-to-archive-a-GitHub-repository-on-Zenodo-at-time-of-publication-1cd6202af5bb4b5ba8464caaba8e9bed) to link your repo to Zenodo and create a new tagged release. (Note: this step is the same one you would follow for a "normal" non-notebook pub.)

    Be sure to use a version number of the form 'v1' or 'v1.0' for this initial tagged release.

1. **Merge the auto-generated publication PR**

    When you create a new tagged release in the step above, a GitHub Action automatically builds the public publication and opens a PR to merge the publication files into the `publish` branch.

    This PR should be merged as-is. If you'd like, you can first preview the final publication by checking out the PR branch locally and running `make preview`.

    The publication is deployed by a second GitHub Action that is triggered when this PR is merged. After a few minutes, the publication should be live and viewable at a URL of the form: `https://arcadiascience.github.io/<your-repo-name>`.

1. **[If necessary] Configure GitHub Pages**

    If you do not see your publication after completing the step above, it is likely because your repository is not configured to host the publication on GitHub Pages.

    To fix this, go to the *Settings* tab of your repository, click on the *Pages* section, and select the "Deploy from a branch" option. Then select the `gh-pages` branch.

## Steps for publishing a revision

The initial steps for publishing a revision are described in the [CONTRIBUTING.qmd](../pages/CONTRIBUTING.qmd) file and are the same for internal and external contributors. These steps culminate in a PR containing changes to the notebook and other files that is reviewed as usual and merged into `main`.

Once such a PR is merged into `main`, you can publish a new version of the pub by creating a new tag and pushing it to the repo.

In your local repo, first make sure you are on the `main` branch and pull the latest changes:

```bash
git checkout main
git pull
```

Then, create a new tag and push it to the repo. For each revision, increment the tag by 1, e.g. 'v1' -> 'v2'. Be sure to use the same format as the initial tagged release (e.g. use 'v2.0' if the initial release was 'v1.0', or 'v2' if the initial release was 'v1').

```bash
git tag v2
git push --tags
```

This will trigger a GitHub Action that builds the publication and opens a PR to merge the publication files into the `publish` branch, just as in the step above ("Create a tagged release of your repo").

Once this PR is merged, the public publication will be updated to include the new version of the notebook (corresponding to the new tag). Previous version(s) of the notebook will remain available via the "versions" menu next to the title of the notebook on the publication page.
