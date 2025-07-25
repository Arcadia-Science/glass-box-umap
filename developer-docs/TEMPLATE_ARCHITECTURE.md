# Template Architecture

## TL;DR for Users

If you're **using this template** to create a publication:

- **Only edit these files**:
  - `index.ipynb` - Your main publication content
  - `_variables.yml` - Publication metadata (title, author, etc.)
  - `authors.yml` - Author information and roles
  - `src/` - Custom Python code for your publication
  - `env.yml` - Dependencies for your publication

- **Leave these files alone**:
  - `_extensions/` - Quarto extensions
  - `_freeze/` - Generated execution results
  - `_site/` - Generated website files
  - `assets/` - Template styling

That's it! Focus on creating great content in your notebook and let the template handle the rest.

## File Structure Overview

```
notebook-pub-template/
├── index.ipynb        # Your main publication content
├── _variables.yml     # Publication metadata
├── authors.yml        # Author information
├── src/               # Custom Python code
├── env.yml            # Conda environment definition
├── _quarto.yml        # Quarto configuration (rarely edit)
├── _freeze/           # Generated output (don't edit)
├── _extensions/       # Quarto extensions (don't edit)
├── _site/             # Generated website (don't edit)
├── assets/            # Template styling (rarely edit)
└── pyproject.toml     # Python configuration (rarely edit)
```

### Content Files

- **`index.ipynb`**: The primary file containing publication content. This Jupyter notebook contains both narrative text (in markdown cells) and executable code (in code cells).

- **`src/`**: Directory for Python modules that contain user code to be imported into your notebook. This helps keep the notebook clean by moving complex functionality into separate files.

### Configuration Files

- **`_variables.yml`**: Contains publication metadata like title, repository information, and Google Analytics ID. This should be customized for the publication.

- **`authors.yml`**: Contains author information and contributor roles. The publishing team will help finalize this file near the end of the publication cycle.

- **`env.yml`**: Defines the Conda environment for the publication.

### Vendored Files

These files should not edited.

- **`_extensions/`**: Contains Quarto extensions that provide special functionality:
  - `mcanouil/iconify/`: Icon rendering
  - `pandoc-ext/abstract-section/`: Used to nicely render the "Summary" in the top markdown cell of the notebook.
  - `quarto-ext/fontawesome/`: Font icons

- **`assets/arcadia.csl`**: This is a custom citation style file built for Arcadia. The details and motivation for this format can be found on [this Notion page](https://www.notion.so/arcadiascience/Changing-citation-styles-on-PubPub-4ad8a40c600f4375b4ffcf1edc77f9a8#4ad8a40c600f4375b4ffcf1edc77f9a8).

### Generated Files

These files are generated and should not be edited directly.

- **`_freeze/`**: Contains cached execution results from the notebook. This should be committed to the repository.

- **`_site/`**: Contains the generated static website. This should never be committed to the repository.

### Template Configuration

- **`_quarto.yml`**: Controls the rendering process and website structure. Only edit when specifically instructed.

- **`assets/css/`**: Stylesheet files that control the publication appearance.

## How It All Works Together

1. **Content Creation**: Authors edit `index.ipynb` with their analysis and narrative
2. **Rendering**: Quarto converts notebooks to HTML using configurations in `_quarto.yml`
3. **Styling**: CSS and HTML snippets in `assets/` apply custom styling to the publication
4. **Output**: Final website is built in `_site/` and execution cache in `_freeze/`
5. **Publishing**: GitHub Actions automate the publication process when merged to the `publish` branch

For more information on how to create and publish content using this template, see:
- [Environment Setup Guide](ENVIRONMENT_SETUP.md)
- [Publishing Guide](PUBLISHING_GUIDE.md)
