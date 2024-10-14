PWD != pwd
BIN = $(PWD)/bin
PATCH_DIR  = $(PWD)/configs

export RESULTS_DIR = results
export REBENCH_DATA = results.data
export PEXECS ?= 30
export ITERS ?= 1

export VENV_DIR = $(PWD)/venv
export PIP = $(VENV_DIR)/bin/pip
export PYTHON_EXEC = $(VENV_DIR)/bin/python
export REBENCH_EXEC = $(VENV_DIR)/bin/rebench
export ALLOY_CFGS = alloy finalise_elide finalise_naive barriers_naive \
	     barriers_none barriers_opt
export ALLOY_DEFAULT_CFG = alloy

export REBENCH_PROCESSOR = $(PWD)/process_graph.py

ALLOY_REPO = https://github.com/jacob-hughes/alloy
ALLOY_SRC_DIR = $(PWD)/alloy
ALLOY_VERSION = master
ALLOY_CFGS_INSTALL_DIRS= $(addprefix $(BIN)/alloy/, $(ALLOY_CFGS))
ALLOY_BOOTSTRAP_STAGE = 1

all: build-alloy clbg awfy sws

.PHONY: build
.PHONY: clean clean-builds check-clean
.PHONY: venv plots

clbg:
	cd clbg_benchmarks && \
		make RUSTC="$(BIN)/alloy/$(ALLOY_DEFAULT_CFG)/bin/rustc"

awfy:
	cd awfy_benchmarks && make

sws:
	cd sws_benchmarks && \
		make RUSTC="$(BIN)/alloy/$(ALLOY_DEFAULT_CFG)/bin/rustc"

plot:
	mkdir -p plots
	cd grmtools_benchmark && make plot
	# cat clbg_benchmarks/summary.csv >> $(PWD)/summary.csv
	# cd awfy_benchmarks && make plot
	# cat awfy_benchmarks/summary.csv >> $(PWD)/summary.csv
	# cd sws_benchmarks && make plot
	# cat sws_benchmarks/summary.csv >> $(PWD)/summary.csv
	# $(PYTHON_EXEC) process_overview.py

bench:
	cd grmtools_benchmark && make bench
	# cd clbg_benchmarks && make bench
	# cd sws_benchmarks && make bench

build: $(ALLOY_CFGS_INSTALL_DIRS)
	cd awfy_benchmarks && make build
	cd clbg_benchmarks && \
		make build RUSTC="$(BIN)/alloy/$(ALLOY_DEFAULT_CFG)/bin/rustc"
	cd sws_benchmarks && \
		make build RUSTC="$(BIN)/alloy/$(ALLOY_DEFAULT_CFG)/bin/rustc"

clean-plots:
	cd clbg_benchmarks && make clean-plots
	cd awfy_benchmarks && make clean-plots
	cd sws_benchmarks && make clean-plots
	- rm summary.csv

clean-benchmarks:
	cd clbg_benchmarks && make clean-benchmarks
	cd awfy_benchmarks && make clean-benchmarks
	cd sws_benchmarks && make clean-benchmarks

clean-builds:
	cd clbg_benchmarks && make clean-builds
	cd awfy_benchmarks && make clean-builds
	cd sws_benchmarks && make clean-builds

build-alloy: venv $(ALLOY_SRC_DIR) $(ALLOY_CFGS_INSTALL_DIRS)

clean: clean-confirm clean-plots clean-benchmarks clean-builds
	rm -rf $(BIN)
	@echo "Clean"

clean-confirm:
	@( read -p "Are you sure? [y/N]: " sure && case "$$sure" in [yY]) true;; *) false;; esac )

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

