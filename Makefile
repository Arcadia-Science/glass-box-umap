.PHONY: lint
lint:
	ruff check --exit-zero .
	ruff format --check .

.PHONY: format
format:
	ruff check --fix .
	ruff format .

.PHONY: pre-commit
pre-commit:
	pre-commit run --all-files

.PHONY: test
test:
	pytest -v .

.PHONY: execute
execute:
	# Project-wide render without executing code cells. Instead, rely on
	# pre-computed results present in _freeze/
	quarto render

	# Now, render `index.ipynb`, with code execution. This will populate
	# _freeze/index/ with pre-computed results.
	quarto render index.ipynb --execute

.PHONY: preview
preview:
	quarto preview

.PHONY: bump-version
bump-version:
	python _bump_version.py
