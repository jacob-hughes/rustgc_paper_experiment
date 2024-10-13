PWD != pwd
BIN = $(PWD)/bin
PATCH_DIR  = $(PWD)/configs

export RESULTS_DIR = results
export REBENCH_DATA = results.data
export PEXECS ?= 30
export ITERS ?= 1

export PYTHON = python3
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

all: bench

.PHONY: build
.PHONY: clean clean-builds
.PHONY: venv plots

plot: plot-clbg plot-awfy plot-sws

bench: clbg awfy sws

clbg:
	cd clbg_benchmarks && \
		make RUSTC="/home/jake/research/alloy/build/x86_64-unknown-linux-gnu/stage1/bin/rustc"
awfy:
	cd awfy_benchmarks && make

sws:
	cd sws_benchmarks && \
		make RUSTC="/home/jake/research/alloy/build/x86_64-unknown-linux-gnu/stage1/bin/rustc"

plot-summary:
	$(PYTHON_EXEC) process_overview.py

clean-plots: clean-clbg-plots clean-awfy-plots clean-sws-plots
	rm summary.csv

clean-clbg-plots:
	cd clbg_benchmarks && make clean-plots

clean-awfy-plots:
	cd awfy_benchmarks && make clean-plots

clean-sws-plots:
	cd sws_benchmarks && make clean-plots

plot-clbg:
	cd clbg_benchmarks && make plot
	cat clbg_benchmarks/summary.csv >> $(PWD)/summary.csv

plot-awfy:
	cd awfy_benchmarks && make plot
	cat awfy_benchmarks/summary.csv >> $(PWD)/summary.csv

plot-sws:
	cd sws_benchmarks && make plot
	cat sws_benchmarks/summary.csv >> $(PWD)/summary.csv

clean-benchmarks: clean-clbg-benchmarks clean-awfy-benchmarks clean-sws-benchmarks

clean-clbg-benchmarks:
	cd clbg_benchmarks && make clean-benchmarks

clean-awfy-benchmarks:
	cd awfy_benchmarks && make clean-benchmarks

clean-sws-benchmarks:
	cd sws_benchmarks && make clean-benchmarks

clean-builds: clean-clbg-builds clean-awfy-builds clean-sws-builds

clean-clbg-builds:
	cd clbg_benchmarks && make clean-builds

clean-awfy-builds:
	cd awfy_benchmarks && make clean-builds

clean-sws-builds:
	cd sws_benchmarks && make clean-builds

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

