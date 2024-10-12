PWD != pwd
PYTHON = python3
BIN = $(PWD)/bin
RESULTS = results

RESULTS_DIR = results
REBENCH_DATA = results.data
export PEXECS ?= 10
export ITERS ?= 5

VENV_DIR = venv
PIP = $(VENV_DIR)/bin/pip
PYTHON_EXEC = $(VENV_DIR)/bin/python
REBENCH_EXEC = $(VENV_DIR)/bin/rebench
PATCH_DIR  = $(PWD)/configs

ALLOY_REPO = https://github.com/jacob-hughes/alloy
ALLOY_SRC_DIR = $(PWD)/alloy
ALLOY_VERSION = master
ALLOY_CFGS = alloy finalise_elide finalise_naive barriers_naive \
	     barriers_none barriers_opt
ALLOY_DEFAULT_CFG = alloy
ALLOY_CFGS_INSTALL_DIRS= $(addprefix $(BIN)/alloy/, $(ALLOY_CFGS))
ALLOY_BOOTSTRAP_STAGE = 1

SOMRS_REPO = https://github.com/Hirevo/som-rs
SOMRS_SRC_DIR = $(PWD)/som-rs
SOMRS_VERSION = 35b780cbee765cca24201fe063d3f1055ec7f608
SOMRS_CFGS = $(addprefix $(BIN)/som-rs/, finalise_elide finalise_naive \
	     barriers_naive barriers_opt barriers_none \
	     perf_rc perf_gc)
SOMRS_EXPS = $(addprefix $(RESULTS)/som-rs/, elision perf barriers)

YKSOM_REPO = https://github.com/softdevteam/yksom
YKSOM_SRC_DIR = $(PWD)/yksom
YKSOM_VERSION=fc7c7c131ba93b7e3c85a172fbcc245f29c324d6
YKSOM_CFGS = finalise_elide finalise_naive barriers_naive \
	     barriers_opt barriers_none

YKSOM_EXPS = $(addprefix $(RESULTS)/yksom/, elision barriers)

YKSOM_CFGS_INSTALL_DIRS = $(addprefix $(PWD)/bin/yksom/, $(YKSOM_CFGS))

# EXPERIMENTS = $(SOMRS_EXPS) $(YKSOM_EXPS)
EXPERIMENTS = $(RESULTS)/regex_redux $(RESULTS)/binary_trees
# EXPERIMENTS = $(RESULTS)/binary_trees


all: bench

.PHONY: build build-som-rs build-yksom build-alloy
.PHONY: clean clean-builds
.PHONY: venv plots

plots:
	@echo $(SOMRS_EXPS)
	mkdir -p plots
	$(PYTHON_EXEC) process_graph.py $(EXPERIMENTS)

bench: $(EXPERIMENTS)

bench-som-rs: $(SOMRS_EXPS)

bench-yksom: $(YKSOM_EXPS)

$(EXPERIMENTS):
	# $(shell basename $(dir $@)).conf $(notdir $@)
	mkdir -p $@
	- $(REBENCH_EXEC) -R -D \
		--invocations ${PEXECS} \
		--iterations ${ITERS} \
		-df $@/$(REBENCH_DATA) \
		regex_redux.conf $(notdir $@)


build: build-som-rs build-yksom


build-binary-trees: $(ALLOY_SRC_DIR)
	mkdir -p $(BIN)/binary_trees/
	cd $(PWD)/clbg && \
	RUSTC="$(BIN)/alloy/$(ALLOY_DEFAULT_CFG)/bin/rustc" \
	      cargo build --release
	ln -s $(PWD)/clbg/target/release/* $(BIN)/binary_trees/

build-yksom: $(YKSOM_CFGS_INSTALL_DIRS)

$(YKSOM_CFGS_INSTALL_DIRS): $(YKSOM_SRC_DIR)/Cargo.toml build-alloy
	cd $(YKSOM_SRC_DIR) && git diff-index --quiet HEAD --
	cd $(YKSOM_SRC_DIR) && git apply $(PATCH_DIR)/yksom/dump_stats.patch
	@if [ -n "$(filter $(ALLOY_CFGS), $(notdir $@))" ]; then \
		cd $(YKSOM_SRC_DIR) && \
		RUSTC="$(BIN)/alloy/$(notdir $@)/bin/rustc" \
			cargo build --release --target-dir=$@; \
	else \
		cd $(YKSOM_SRC_DIR) && \
		RUSTC="$(BIN)/alloy/$(ALLOY_DEFAULT_CFG)/bin/rustc" \
			cargo build --release --target-dir=$@; \
	fi
	ln -s $(YKSOM_SRC_DIR)/SOM $@/SOM
	ln -s $@/release/yksom $@/yksom
	cd $(YKSOM_SRC_DIR) && git reset --hard

$(YKSOM_SRC_DIR)/Cargo.toml:
	git clone $(YKSOM_REPO) $(YKSOM_SRC_DIR)
	cd $(YKSOM_SRC_DIR) && git checkout $(YKSOM_VERSION)

build-som-rs: $(SOMRS_CFGS)

$(SOMRS_CFGS): $(SOMRS_SRC_DIR)/Cargo.lock $(ALLOY_CFGS_INSTALL_DIRS)
	cd $(SOMRS_SRC_DIR) && git diff-index --quiet HEAD --
	@if [  "$(notdir $@))" = "perf_rc" ]; then \
		cd $(SOMRS_SRC_DIR) && git apply $(PATCH_DIR)/som-rs/bdwgc_allocator.patch; \
	else \
		cd $(SOMRS_SRC_DIR) && git apply $(PATCH_DIR)/som-rs/use_gc.patch; \
		cd $(SOMRS_SRC_DIR) && git apply $(PATCH_DIR)/som-rs/dump_stats.patch; \
	fi
	@if [ -n "$(filter $(ALLOY_CFGS), $(notdir $@))" ]; then \
		cd $(SOMRS_SRC_DIR) && \
		RUSTC="$(BIN)/alloy/$(notdir $@)/bin/rustc" \
			cargo build \
				--release -p som-interpreter-bc \
			      	--target-dir=$@; \
	else \
		cd $(SOMRS_SRC_DIR) && \
		RUSTC="$(BIN)/alloy/$(ALLOY_DEFAULT_CFG)/bin/rustc" \
			cargo build \
				--release -p som-interpreter-bc \
			      	--target-dir=$@; \
	fi
	ln -s $(SOMRS_SRC_DIR)/core-lib $@/core-lib
	ln -s $@/release/som-interpreter-bc $@/som-rs
	cd $(SOMRS_SRC_DIR) && git reset --hard

$(SOMRS_SRC_DIR)/Cargo.lock:
	git clone --recursive $(SOMRS_REPO) $(SOMRS_SRC_DIR)
	cd $(SOMRS_SRC_DIR) && git checkout $(SOMRS_VERSION)

build-alloy: $(ALLOY_CFGS_INSTALL_DIRS)

$(ALLOY_CFGS_INSTALL_DIRS):
	cd $(ALLOY_SRC_DIR) && git diff-index --quiet HEAD --
	@if [ -f "$(PATCH_DIR)/alloy/$(notdir $@).patch" ]; then \
		cd $(ALLOY_SRC_DIR) && git apply $(PATCH_DIR)/alloy/$(notdir $@).patch; \
	fi
	$(PYTHON) $(ALLOY_SRC_DIR)/x.py install --config benchmark.config.toml \
		--stage $(ALLOY_BOOTSTRAP_STAGE) \
		--build-dir $(ALLOY_SRC_DIR)/build \
		--set build.docs=false \
		--set install.prefix=$@ \
		--set install.sysconfdir=etc
	cd $(ALLOY_SRC_DIR) && git reset --hard

$(ALLOY_SRC_DIR):
	git clone $(ALLOY_REPO) $(ALLOY_SRC_DIR)
	cd $(ALLOY_SRC_DIR) && git checkout $(ALLOY_VERSION)

venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install -r requirements.txt

clean-benchmarks:
	rm -rf ${RESULTS_DIR}

clean-builds:
	rm -rf bin

clean: clean-benchmarks
