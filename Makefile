PWD != pwd
BIN = $(PWD)/bin
PATCH_DIR  = $(PWD)/configs

PYTHON = python3

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

ALLOY_REPO = https://github.com/softdevteam/alloy
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
	# cd grmtools_benchmarks && make plot
	# cat grmtools_benchmarks/summary.csv >> $(PWD)/summary.csv
	# cat clbg_benchmarks/summary.csv >> $(PWD)/summary.csv
	cd awfy_benchmarks && make plot
	# cat awfy_benchmarks/summary.csv >> $(PWD)/summary.csv
	# cd sws_benchmarks && make plot
	# cat sws_benchmarks/summary.csv >> $(PWD)/summary.csv
	# $(PYTHON_EXEC) process_overview.py

bench:
	# cd grmtools_benchmarks && make bench
	# cd clbg_benchmarks && make bench
	cd awfy_benchmarks && make bench
	# cd sws_benchmarks && make bench

build: build-alloy
	cd awfy_benchmarks && make build
	# cd clbg_benchmarks && \
	# 	make build RUSTC="$(BIN)/alloy/$(ALLOY_DEFAULT_CFG)/bin/rustc"
	# cd sws_benchmarks && \
	# 	make build RUSTC="$(BIN)/alloy/$(ALLOY_DEFAULT_CFG)/bin/rustc"
	# cd grmtools_benchmarks && \
	# 	make build RUSTC="$(BIN)/alloy/$(ALLOY_DEFAULT_CFG)/bin/rustc"

clean-plots:
	cd clbg_benchmarks && make clean-plots
	cd awfy_benchmarks && make clean-plots
	cd sws_benchmarks && make clean-plots
	cd grmtools_benchmarks && make clean-plots
	- rm summary.csv

clean-benchmarks:
	cd clbg_benchmarks && make clean-benchmarks
	cd awfy_benchmarks && make clean-benchmarks
	cd sws_benchmarks && make clean-benchmarks

clean-builds:
	cd clbg_benchmarks && make clean-builds
	cd awfy_benchmarks && make clean-builds
	cd sws_benchmarks && make clean-builds


build-alloy: venv $(ALLOY_SRC_DIR) build-alloy-barriers-none \
	build-alloy-barriers-naive build-alloy-barriers-opt \
	build-alloy-finalise-elide build-alloy-finalise-naive

build-alloy-barriers-naive:
	cd $(ALLOY_SRC_DIR) && git reset --hard && ./x.py clean
	cd $(ALLOY_SRC_DIR) && git apply $(PATCH_DIR)/alloy/barriers_naive.patch
	$(PYTHON) $(ALLOY_SRC_DIR)/x.py install --config benchmark.config.toml \
		--stage $(ALLOY_BOOTSTRAP_STAGE) \
		--build-dir $(ALLOY_SRC_DIR)/build \
		--set build.docs=false \
		--set install.prefix=${BIN}/alloy/barriers_naive \
		--set install.sysconfdir=etc

build-alloy-barriers-opt:
	cd $(ALLOY_SRC_DIR) && git reset --hard && ./x.py clean
	$(PYTHON) $(ALLOY_SRC_DIR)/x.py install --config benchmark.config.toml \
		--stage $(ALLOY_BOOTSTRAP_STAGE) \
		--build-dir $(ALLOY_SRC_DIR)/build \
		--set build.docs=false \
		--set install.prefix=${BIN}/alloy/barriers_opt \
		--set install.sysconfdir=etc

build-alloy-barriers-none:
	cd $(ALLOY_SRC_DIR) && git reset --hard && ./x.py clean
	cd $(ALLOY_SRC_DIR) && git apply $(PATCH_DIR)/alloy/barriers_none.patch
	$(PYTHON) $(ALLOY_SRC_DIR)/x.py install --config benchmark.config.toml \
		--stage $(ALLOY_BOOTSTRAP_STAGE) \
		--build-dir $(ALLOY_SRC_DIR)/build \
		--set build.docs=false \
		--set install.prefix=${BIN}/alloy/barriers_none \
		--set install.sysconfdir=etc

build-alloy-finalise-elide:
	cd $(ALLOY_SRC_DIR) && git reset --hard && ./x.py clean
	$(PYTHON) $(ALLOY_SRC_DIR)/x.py install --config benchmark.config.toml \
		--stage $(ALLOY_BOOTSTRAP_STAGE) \
		--build-dir $(ALLOY_SRC_DIR)/build \
		--set build.docs=false \
		--set install.prefix=${BIN}/alloy/finalise_elide \
		--set install.sysconfdir=etc

build-alloy-finalise-naive:
	cd $(ALLOY_SRC_DIR) && git reset --hard && ./x.py clean
	cd $(ALLOY_SRC_DIR) && git apply $(PATCH_DIR)/alloy/finalise_naive.patch
	$(PYTHON) $(ALLOY_SRC_DIR)/x.py install --config benchmark.config.toml \
		--stage $(ALLOY_BOOTSTRAP_STAGE) \
		--build-dir $(ALLOY_SRC_DIR)/build \
		--set build.docs=false \
		--set install.prefix=${BIN}/alloy/finalise_naive \
		--set install.sysconfdir=etc

clean: clean-confirm clean-plots clean-benchmarks clean-builds
	rm -rf $(BIN)
	@echo "Clean"

clean-confirm:
	@( read -p "Are you sure? [y/N]: " sure && case "$$sure" in [yY]) true;; *) false;; esac )

$(ALLOY_SRC_DIR):
	git clone $(ALLOY_REPO) $(ALLOY_SRC_DIR)
	cd $(ALLOY_SRC_DIR) && git checkout $(ALLOY_VERSION)

venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install -r requirements.txt

