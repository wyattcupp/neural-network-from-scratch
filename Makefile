# ===================================================
# Makefile for building neural_network package.
# Wyatt Cupp <wyattcupp@gmail.com>
# ===================================================
#
# Configuration
# =============
GIT       :=git
PYTHON    :=python3
DIST_DIR  :=out
BUILD_DIR :=.
PYLINT    :=pylint
VENV	  :=.venv

TEST_PREFIX             :=test
TEST_DIR                := tests/

# Targets
# =======
## help: Output this message and exit.
help:
	@echo '================================================'
	@echo '                   Build Targets'
	@echo '================================================'
	@echo
	@fgrep -h '##' $(MAKEFILE_LIST) | fgrep -v fgrep | column -t -s ':' | sed -e 's/## //'
.PHONY: help

## all: Cleans, builds, lints, and unit-tests the library.
all: clean build lint test

## build: Build the neural_network package locally
.PHONY: build
build: clean
	if [ ! -d $(VENV) ] && [ ! -L $(VENV) ]; then \
		python -m venv $(VENV); \
	fi
	source $(VENV)/bin/activate && pip install .

## build-dev: Builds the library using an editable developer installation of the neural-network library
.PHONY: build-dev
build-dev: clean
	if [ ! -d $(VENV) ] && [ ! -L $(VENV) ]; then \
		python -m venv $(VENV); \
	fi
	source $(VENV)/bin/activate && pip install -e .

## test: Runs all configured unit test cases
.PHONY: test
test:
	if [ -d $(VENV) ]; then \
		source $(VENV)/bin/activate; \
		pip install pytest; \
		pytest -v $(TEST_DIR); \
	else echo "Please create a virtual environment via [python -m venv .venv]"; \
	fi

## lint: Execute lint (pylint) on current library codebase
.PHONY: lint
lint:
	if [ ! -d $(VENV) ] && [ ! -L $(VENV) ]; then \
		echo "Run [make build] and then try again."; \
		exit 1; \
	fi

	source $(VENV)/bin/activate && pip install pylint && $(PYLINT) ./neural_network && $(PYLINT) ./tests

# clean: Removes /.venv/, uninstalls neural-network, etc.
.PHONY: clean
clean:
	rm -rf .venv
	find ../neural-network -type f -name "*.pyc" -exec rm -rf {} \;
	find ../neural-network -type d -name "build" -prune -exec rm -rf {} \;
	find ../neural-network -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} \;
	find ../neural-network -type d -name "__pycache__" -prune -exec rm -rf {} \;
	find ../neural-network -type d -name ".pytest_cache" -prune -exec rm -rf {} \;
	find ../neural-network -type d -name "neural_network.egg-info" -prune -exec rm -rf {} \;