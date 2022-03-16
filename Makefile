# Detect platform
ifeq ($(OS), Windows_NT)
	PLATFORM := win-64
else
	uname := $(shell sh -c 'uname 2>/dev/null || echo unknown')
	ifeq ($(uname), Darwin)
		PLATFORM := osx-64
	else ifeq ($(uname), Linux)
		PLATFORM := linux-64
	else
		@echo "Unsupported platform"
		exit 1
	endif
endif

all:
	@echo "Detected platform: $(PLATFORM)"

# -- Dependency management with Pip --------------------------------------------

# Update packaging tools
pip-update-tools:
	pip install --upgrade pip-tools pip setuptools

# Update .in files
pip-update-in-files:
	python3 requirements/make_pip_in_files.py --quiet

# Lock pip dependencies
# Dev must be compiled first because it constrains the others
# No hashes: doesn't play nicely with RTD when running pip-compile on macOS
pip-compile: pip-update-in-files
	rm requirements/dev.txt
	touch requirements/dev.txt

	@for LAYER in dev main docs tests; do \
		echo "Compiling requirements/$${LAYER}.in to requirements/$${LAYER}.txt"; \
		pip-compile --upgrade --build-isolation --allow-unsafe \
			--output-file requirements/$${LAYER}.txt \
			requirements/$${LAYER}.in; \
	done

# Lock dependencies
pip-lock: pip-update-tools pip-compile

# Initialise development environment
pip-init:
	pip install --upgrade -r requirements/dev.txt
	pip install --editable . --no-deps

pip-update: pip-lock pip-init

.PHONY: pip-compile pip-update-tools pip-update-deps pip-init pip-update-in-files

# -- Dependency management with Conda ------------------------------------------

# Lock conda dependencies
conda-lock:
	python3 requirements/make_conda_env.py -o requirements/environment.yml --quiet
	conda-lock --kind explicit --mamba --file requirements/environment.yml \
	    --filename-template "requirements/environment-{platform}.lock" \
	    -p $(PLATFORM)

conda-lock-all:
	python3 requirements/make_conda_env.py -o requirements/environment.yml --quiet
	conda-lock --kind explicit --mamba --file requirements/environment.yml \
	    --filename-template "requirements/environment-{platform}.lock" \
	    -p osx-64 -p linux-64

# Initialise development environment
conda-init:
	python3 requirements/check_conda_env.py
	conda config --env --add channels conda-forge --add channels eradiate
	conda update --file requirements/environment-$(PLATFORM).lock
	python3 requirements/copy_envvars.py
	pip install --editable . --no-deps

conda-update: conda-lock-all conda-init

.PHONY: conda-lock conda-lock-all conda-init conda-update

# -- Documentation -------------------------------------------------------------

.PHONY: docs docs-html-plot

docs:
	make -C docs html
	@echo "Documentation index at docs/_build/html/index.html"

docs-serve:
	make -C docs serve

docs-linkcheck:
	make -C docs linkcheck

docs-rst:
	make -C docs rst-api
	make -C docs rst-plugins

docs-clean:
	make -C docs clean

# -- Testing -------------------------------------------------------------------

.PHONY: pytest pytest-slow pytest-notslow pytest-formatters

pytest:
	pytest tests

pytest-slow:
	pytest -m "slow" tests

pytest-notslow:
	pytest -m "not slow" tests

pytest-formatters:
	pytest --black --isort src/eradiate
