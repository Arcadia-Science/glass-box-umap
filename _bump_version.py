"""Creates a version stamp of the current `index.ipynb`.

* Copies `index.ipynb` to `index_v??.ipynb`, where ?? is the latest version number prior
  to running this script.
* Copies _freeze/index to _freeze/index_??
* Adds v?? to the the version navbar at the top of the site.

Important:
    - You should not run this command directly. Use `make bump-version`, and run it only
      when instructed to in `CONTRIBUTING.qmd`.
"""

import shutil
import sys
from pathlib import Path
from typing import Any

import yaml


def extract_current_version(yaml_content: dict[str, Any]) -> int | None:
    """Extract the current version from the YAML menu items."""
    website: dict[str, Any] = yaml_content.get("website", {})
    navbar: dict[str, Any] = website.get("navbar", {})
    left: list[dict[str, Any]] = navbar.get("left", [])

    for item in left:
        if "version-control" in item.get("text", ""):
            menu: list[dict[str, str]] = item.get("menu", [])
            for menu_item in menu:
                if "(latest)" in menu_item.get("text", ""):
                    version: int = int(menu_item["text"].split()[0][1:])
                    return version


def update_quarto_yaml(path: Path) -> tuple[int, int]:
    """Update the _quarto.yml file with new version information.

    Args:
        file_path: Path to the YAML file

    Returns:
        Tuple of (current_version, next_version)

    Raises:
        ValueError: If current version cannot be found in YAML
    """
    content: dict[str, Any] = yaml.safe_load(path.read_text())

    current_version: int | None = extract_current_version(content)
    if current_version is None:
        raise ValueError("Could not find current version in YAML")

    next_version: int = current_version + 1

    for item in content["website"]["navbar"]["left"]:
        if "version-control" in item.get("text", ""):
            menu: list[dict[str, str]] = item["menu"]
            new_latest: dict[str, str] = {
                "text": f"v{next_version:02d} (latest)",
                "href": "index.ipynb",
            }
            archived: dict[str, str] = {
                "text": f"v{current_version:02d}",
                "href": f"index_v{current_version:02d}.ipynb",
            }
            # Insert new entries into top of menu list
            item["menu"] = [new_latest, archived] + menu[1:]

    path.write_text(yaml.dump(content, sort_keys=False, allow_unicode=True))
    return current_version, next_version


def copy_files(current_version: int) -> None:
    """Copy the index files to versioned copies."""
    src_notebook: Path = Path("index.ipynb")
    dst_notebook: Path = Path(f"index_v{current_version:02d}.ipynb")
    shutil.copy2(src_notebook, dst_notebook)

    src_freeze: Path = Path("_freeze/index")

    dst_freeze: Path = Path(f"_freeze/index_v{current_version:02d}")
    if dst_freeze.exists():
        shutil.rmtree(dst_freeze)

    shutil.copytree(src_freeze, dst_freeze)


def main() -> None:
    """Main entry point for the version bump script."""
    yaml_file: Path = Path("_quarto.yml")

    try:
        current_version, next_version = update_quarto_yaml(yaml_file)
        print(f"Updated YAML: v{current_version:02d} -> v{next_version:02d}")
        copy_files(current_version)
        print(f"Copied files for v{current_version:02d}")
        print("Version bump completed successfully!")
    except Exception as e:
        print(f"Error during version bump. Unclean git state may have been created. Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
