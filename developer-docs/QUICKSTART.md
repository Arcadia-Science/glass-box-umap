## Quick Start

1. **Create a repo from this template**

    In the top-right of this GitHub repo, select the green button that says "*Use this template*".

    **IMPORTANT**: When creating your repo from this template, **check the box** that says, "*Include all branches*". This is required because the hosted publication is managed on a separate branch.

2. **Configure your publication**

    * Replace the variables in `_variables.yml`
      - For `google_analytics_id`, contact the Publishing Team with your repo name to receive a tracking ID
    * Feel free to edit the variables in `authors.yml`
      - Ultimately, Publishing Team will provide you with an `authors.yml` based on the contributor roles they assign for the publication, so this isn't necessary

3. **Install Quarto**

    The publication is rendered with [Quarto](https://quarto.org/). If you don't have it installed (check with `quarto --version`), you can [install it here](https://quarto.org/docs/get-started/).

4. **Set up your environment**

    See the [Environment Setup Guide](developer-docs/ENVIRONMENT_SETUP.md) for complete instructions.

5. **Register your publication**

    If you intend to publish your analysis, fill out the "*Kick off a new pub*" form on the AirTable [Publishing toolkit](https://www.notion.so/arcadiascience/Publishing-2-0-f0c51bf29d1d4356a86e6cf8a72ae88b?pvs=4#e1de83e8dd2a4081904064347779ed25).

6. **Create your publication**

    Edit `index.ipynb` to create your publication. As you work, you can render a live preview of your changes with:

    ```bash
    make preview
    ```

    Then, commit your changes to a development branch and merge them into `main` using our usual PR-based workflow.

    As you work, please be careful to avoid modifying any files in the following directories:

      - `_extensions/` (Quarto extensions)
      - `_freeze/` (Generated execution results)
      - `_site/` (Generated website files)
      - `assets/` (Template styling)

    These files are all either necessary to build the publication or are automatically generated during the publication process.

7. **Publishing**

    See the [Publishing Guide](developer-docs/PUBLISHING_GUIDE.md) for complete instructions on the publishing process.

