CURDIR ?= $(shell pwd)

IMAGE := $(if $(FULL),full:latest,quick:latest)
LOG_STAGE := log_export
RUNTIME_STAGE := runtime
RESULTS := ./results
TABLES := ./tables
PLOTS := ./plots
LOGFILE := $(CURDIR)/experiment.log

VENV_DIR := $(CURDIR)/.venv
VENV_PYTHON := $(VENV_DIR)/bin/python
PYTHON := $(VENV_PYTHON)
INVOKE := $(VENV_DIR)/bin/invoke

BIN_DIR ?= $(CURDIR)/artefacts/bin
BIN_ARCHIVE ?= artefacts-bin.tar.xz
GDRIVE_FID = 1hwNZbAEEJPoFkvYoq-J4yadFsbsnbHdU
GDOWN = $(VENV_DIR)/bin/gdown


EXP_ARG := $(if $(strip $(EXPERIMENTS)),--experiments "$(EXPERIMENTS)")
SUITE_ARG := $(if $(strip $(SUITES)),--suites "$(SUITES)")
MEASURE_ARG := $(if $(strip $(MEASUREMENTS)),--measurements $(MEASUREMENTS))
QUICK_PEXECS = 5
FULL_PEXECS = 30

.PHONY: run-full run-quick fetch-binaries bare-metal

$(VENV_DIR):
	python3 -m venv $(VENV_DIR)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

venv: pyproject.toml | $(VENV_DIR)


define WITH_DOCKER
	docker buildx build --progress=plain \
		--build-arg FULL=$1 \
		$(if $(EXP_ARG),--build-arg EXPERIMENTS='$(EXPERIMENTS)') \
		$(if $(SUITE_ARG),--build-arg SUITES='$(SUITES)') \
		$(if $(MEASURE_ARG),--build-arg MEASUREMENTS='$(MEASUREMENTS)') \
		--target runtime \
		--tag $2 \
		--load .
	docker run --rm -it \
		--mount type=bind,source="$(RESULTS)",target=/app/results \
		--mount type=bind,source="$(TABLES)",target=/app/tables \
		--mount type=bind,source="$(PLOTS)",target=/app/plots \
		$2
endef

results:
	mkdir results

plots:
	mkdir plots

tables:
	mkdir tables

run-quick: results plots tables
	$(call WITH_DOCKER,false,quick:latest)

run-full: results plots tables
	$(call WITH_DOCKER,true,full:latest)

tarball:
	@echo "Compressing $(BIN_DIR) with maximum compression..."
	cd artefacts && tar -cf - bin | xz -9e > $(BIN_ARCHIVE)
	@echo "Created $(BIN_ARCHIVE) ($$(du -h artefacts/$(BIN_ARCHIVE) | cut -f1))"

fetch-binaries: $(GDRIVE_ARTEFACT)

$(GDRIVE_ARTEFACT):
	./fetch_binaries.sh

bare-metal: venv
	@$(INVOKE) build-benchmarks \
		$(EXP_ARG) \
		$(SUITE_ARG) \
		$(MEASURE_ARG)
	@$(INVOKE) run-benchmarks \
		$(if $(PEXECS),$(PEXECS),$(FULL_PEXECS)) \
		$(EXP_ARG) \
		$(SUITE_ARG) \
		$(MEASURE_ARG)

run-benchmarks: venv
	@$(INVOKE) run-benchmarks \
		$(if $(PEXECS),$(PEXECS),$(FULL_PEXECS)) \
		$(EXP_ARG) \
		$(SUITE_ARG) \
		$(MEASURE_ARG)

build-benchmarks: venv
	@$(INVOKE) build-benchmarks \
		$(EXP_ARG) \
		$(SUITE_ARG) \
		$(MEASURE_ARG)

build-alloy: venv
	@$(INVOKE) build-alloy $(EXP_ARG)

download-bins: venv
	@$(GDOWN) $(GDRIVE_FID) -O $(CURDIR)/artefacts-bin.tar.xz
	mkdir -p artefacts
	tar -xvf artefacts-bin.tar.xz -C artefacts
	$(INVOKE) prerequisites

build-heaptrack: venv
	@$(INVOKE) build-heaptrack

process: venv
	@$(INVOKE) process-results $(EXP_ARG) $(SUITE_ARG) $(MEASURE_ARG)

build-paper:
	docker build --progress=plain -f Dockerfile.paper --target export --output type=local,dest=./rustgc_paper -t rustgc-paper:latest .
