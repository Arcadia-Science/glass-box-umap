# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pyyaml>=6.0.2",
# ]
# ///

"""
This script gathers all of the files needed to build the publication.
These files do not all exist in any single commit in the repository,
because the publication needs to include versions of the `index.ipynb` notebook
for each git tag.

This script executes the following steps:
- Copies the `index.ipynb` notebook from each tag to a `_freeze/index_v{tag}.ipynb` file.
- Copies the `_freeze/index` directory from each tag to a `_freeze/index_v{tag}` directory.
- Updates the `_quarto.yml` file to include a `version-control` item for each tag.
"""

import argparse
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

import yaml


def get_versioned_notebook_path(tag: str) -> Path:
    """Get the path to the versioned `index.ipynb` notebook for a given tag."""
    return Path(f"index_{tag}.ipynb")


def get_versioned_freeze_directory_path(tag: str) -> Path:
    """Get the path to the versioned `_freeze/index` directory for a given tag."""
    return Path(f"_freeze/index_{tag}")


@contextmanager
def git_checkout(ref: str):
    original_ref = (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode("utf-8")
        .strip()
    )

    try:
        subprocess.run(["git", "checkout", ref], check=True)
        yield
    finally:
        subprocess.run(["git", "checkout", original_ref], check=True)


def get_tags() -> list[str]:
    """Get a list of all git tags."""
    result = subprocess.run(["git", "tag"], capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


def copy_notebook(tag: str, dry_run: bool) -> None:
    """Copy the `index.ipynb` notebook for a given tag to a versioned notebook."""
    src = Path("index.ipynb")
    dst = get_versioned_notebook_path(tag)

    if dry_run:
        print(f"Would copy '{src}' to '{dst}'")
        return

    shutil.copy2(src, dst)


def copy_freeze_directory(tag: str, dry_run: bool) -> None:
    """Copy the `_freeze/index` directory for a given tag to a versioned directory."""
    src = Path("_freeze/index")
    dst = get_versioned_freeze_directory_path(tag)

    if dry_run:
        print(f"Would copy '{src}' to '{dst}'")
        return

    shutil.copytree(src, dst, dirs_exist_ok=True)


def update_index_notebook_and_freeze_directory(tag: str, dry_run: bool) -> None:
    """
    Rename the most recent tagged version of the notebook and its freeze directory
    to `index.ipynb` and `_freeze/index`, respectively.
    """
    notebook_src = get_versioned_notebook_path(tag)
    notebook_dst = Path("index.ipynb")

    freeze_src = get_versioned_freeze_directory_path(tag)
    freeze_dst = Path("_freeze/index")

    if dry_run:
        print(f"Would move '{notebook_src}' to '{notebook_dst}'")
        print(f"Would move '{freeze_src}' to '{freeze_dst}'")
        return

    notebook_dst.unlink()
    notebook_src.replace(notebook_dst)

    shutil.rmtree(freeze_dst, ignore_errors=True)
    freeze_src.replace(freeze_dst)


def update_quarto_yaml(most_recent_tag: str, previous_tags: list[str], dry_run: bool) -> None:
    """Update the _quarto.yml file to include menu items for each tagged release."""
    yaml_path = Path("_quarto.yml")
    content = yaml.safe_load(yaml_path.read_text())

    most_recent_version_item = {
        "text": f"{most_recent_tag} (latest)",
        # The link for the most recent version is always `index.ipynb`.
        # (This version of the notebook is renamed to `index.ipynb` elsewhere in this script.)
        "href": "index.ipynb",
    }
    previous_version_items = [
        {"text": tag, "href": str(get_versioned_notebook_path(tag))} for tag in previous_tags
    ]

    if dry_run:
        print(f"Would update '{yaml_path}' with the following menu items:")
        print(f"  - {most_recent_version_item}")
        for item in previous_version_items:
            print(f"  - {item}")
        return

    # Insert the new items.
    for item in content["website"]["navbar"]["left"]:
        if "version-control" in item.get("text", ""):
            item["menu"] = [most_recent_version_item] + previous_version_items

    yaml_path.write_text(yaml.dump(content, sort_keys=False, allow_unicode=True))


def main() -> None:
    """
    Entrypoint for the script.

    CLI args:
        --dry-run: Print the actions that would be taken without actually taking them.
            This is intended to be used when running this script locally during development,
            as a way to preview the changes the script would make to the working directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tags = get_tags()
    if not tags:
        raise ValueError("No tags found")

    for tag in tags:
        print(f"Processing tag {tag}")
        with git_checkout(tag):
            copy_notebook(tag, dry_run=args.dry_run)
            copy_freeze_directory(tag, dry_run=args.dry_run)

    tags = sorted(tags, reverse=True)
    most_recent_tag, *previous_tags = tags

    update_index_notebook_and_freeze_directory(most_recent_tag, dry_run=args.dry_run)
    update_quarto_yaml(most_recent_tag, previous_tags, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
