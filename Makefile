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
	python requirements/make_pip_in_files.py --quiet

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
	python setup.py develop

pip-update: pip-lock pip-init

.PHONY: pip-compile pip-update-tools pip-update-deps pip-init pip-update-in-files

# -- Dependency management with Conda ------------------------------------------

# Lock conda dependencies
conda-lock:
	python requirements/make_conda_env.py -o requirements/environment.yml --quiet
	conda-lock --file requirements/environment.yml \
	    --filename-template "requirements/environment-{platform}.lock" \
	    -p $(PLATFORM)

conda-lock-all:
	python requirements/make_conda_env.py -o requirements/environment.yml --quiet
	conda-lock --file requirements/environment.yml \
	    --filename-template "requirements/environment-{platform}.lock" \
	    -p osx-64 -p linux-64

# Initialise development environment
conda-init:
	python requirements/check_conda_env.py
	conda config --env --add channels conda-forge --add channels eradiate
	conda update --file requirements/environment-$(PLATFORM).lock
	python requirements/copy_envvars.py
	python setup.py develop

conda-update: conda-lock-all conda-init

.PHONY: conda-lock conda-lock-all conda-init conda-update

# -- Testing -------------------------------------------------------------------

.PHONY: pytest pytest-slow pytest-notslow pytest-formatters

pytest:
	pytest eradiate

pytest-slow:
	pytest -m "slow" eradiate

pytest-notslow:
	pytest -m "not slow" eradiate

pytest-formatters:
	pytest --black --isort --ignore-glob="*test_*.py" eradiate