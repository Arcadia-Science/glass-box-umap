DOCS_DIR := ./docs

# Load environment variables from the `.env` file if it exists.
ifneq (,$(wildcard .env))
    include .env
endif

.PHONY: lint
lint:
	ruff check --exit-zero .
	ruff format --check .

.PHONY: format
format:
	ruff format .
	ruff check --fix .

.PHONY: typecheck
typecheck:
	uv run pyright --project pyproject.toml src/ tests/

.PHONY: pre-commit
pre-commit:
	uv run pre-commit run --all-files

.PHONY: test
test:
	uv run pytest -v .

.PHONY: clean
clean:
	rm -rf dist

.PHONY: build
build: clean
	uv build

.PHONY: notebooks
notebooks:
	uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 docs/user_guide/*.ipynb
	uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 docs/examples/*.ipynb

.PHONY: docs
docs:
	$(MAKE) -C docs/ clean-and-build-html
	$(MAKE) -C docs/ view-html

.PHONY: docs-live
docs-live:
	$(MAKE) -C docs/ clean-and-build-html
	$(MAKE) -C docs/ live

.PHONY: docs-with-notebooks
docs-with-notebooks: notebooks docs

.PHONY: build-and-test-publish
build-and-test-publish: build
	uv publish \
		--publish-url https://test.pypi.org/legacy/ \
		--token ${UV_PUBLISH_TOKEN_TEST}

.PHONY: build-and-publish
build-and-publish: build
	uv publish \
		--token ${UV_PUBLISH_TOKEN}
