PWD != pwd
BIN = $(PWD)/bin

# VENV
PYTHON = python3
VENV_DIR = $(PWD)/venv
PIP = $(VENV_DIR)/bin/pip
PYTHON_EXEC = $(VENV_DIR)/bin/python
export REBENCH_EXEC = $(VENV_DIR)/bin/rebench

export REBENCH_PROCESSOR = $(PWD)/process_graph.py

# BUILD
ALLOY_REPO = https://github.com/jacob-hughes/alloy
ALLOY_SRC_DIR = $(PWD)/alloy
ALLOY_VERSION = handle_intrinsics
ALLOY_BOOTSTRAP_STAGE = 1
ALLOY_CFGS = alloy finalise_elide finalise_naive barriers_naive \
	     barriers_none barriers_opt
PATCHES  = $(PWD)/patches

# BENCH
export PEXECS ?= 10
# Only applicable for AWFY and CLBG
export ITERS ?= 1

all: build bench plot

.PHONY: venv
.PHONY: build build-alloy build-perf build-barriers build-elision
.PHONY: bench bench-perf bench-barriers bench-elision
.PHONY: plot plot-perf plot-barriers plot-elision
.PHONY: clean clean-builds clean-plots clean-benchmarks check-clean

plot: plot-perf plot-barriers plot-elision

plot-perf:
	cd sws_benchmarks && make plot-perf
	cd awfy_benchmarks && make plot-perf
	cd grmtools_benchmarks && make plot-perf

plot-barriers:
	cd sws_benchmarks && make plot-barriers
	cd awfy_benchmarks && make plot-barriers
	cd grmtools_benchmarks && make plot-barriers

plot-elision:
	cd sws_benchmarks && make plot-elision
	cd awfy_benchmarks && make plot-elision
	cd grmtools_benchmarks && make plot-elision

bench: bench-perf bench-barriers bench-elision

bench-perf:
	cd sws_benchmarks && make bench-perf
	cd awfy_benchmarks && make bench-perf
	cd grmtools_benchmarks && make bench-perf

bench-barriers:
	cd sws_benchmarks && make bench-barriers
	cd awfy_benchmarks && make bench-barriers
	cd grmtools_benchmarks && make bench-barriers

bench-elision:
	cd sws_benchmarks && make bench-elision
	cd awfy_benchmarks && make bench-elision
	cd grmtools_benchmarks && make bench-elision

build: build-alloy build-benchmarks

build-benchmarks: build-perf build-barriers build-elision

build-perf:
	cd sws_benchmarks && make build-perf
	cd awfy_benchmarks && make build-perf
	cd grmtools_benchmarks && make build-perf

build-barriers:
	cd sws_benchmarks && make build-barriers
	cd awfy_benchmarks && make build-barriers
	cd grmtools_benchmarks && make build-barriers

build-elision:
	cd sws_benchmarks && make build-elision
	cd awfy_benchmarks && make build-elision
	cd grmtools_benchmarks && make build-elision

build-alloy: $(addprefix $(BIN)/alloy/, $(ALLOY_CFGS)) $(ALLOY_SRC_DIR)

$(addprefix $(BIN)/alloy/, $(ALLOY_CFGS)): $(ALLOY_SRC_DIR)
	cd $(ALLOY_SRC_DIR) && git reset --hard && ./x.py clean
	if [ -f "$(PATCHES)/alloy/$(notdir $@).patch" ]; then \
		echo "git apply $(PATCHES)/alloy/$(notdir $@).patch"; \
		cd $(ALLOY_SRC_DIR) && git apply $(PATCHES)/alloy/$(notdir $@).patch; \
	fi
	$(PYTHON) $(ALLOY_SRC_DIR)/x.py install --config benchmark.config.toml \
		--stage $(ALLOY_BOOTSTRAP_STAGE) \
		--build-dir $(ALLOY_SRC_DIR)/build \
		--set build.docs=false \
		--set install.prefix=$@ \
		--set install.sysconfdir=etc

$(ALLOY_SRC_DIR): venv
	git clone $(ALLOY_REPO) $(ALLOY_SRC_DIR)
	cd $(ALLOY_SRC_DIR) && git checkout $(ALLOY_VERSION)

venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install -r requirements.txt

clean: clean-confirm clean-plots clean-benchmarks clean-builds
	rm -rf $(ALLOY_SRC_DIR)
	@echo "Clean"

clean-confirm:
	@echo $@
	@( read -p "Are you sure? [y/N]: " sure && case "$$sure" in [yY]) true;; *) false;; esac )

clean-plots:
	cd sws_benchmarks && make clean-plots
	cd awfy_benchmarks && make clean-plots
	cd grmtools_benchmarks && make clean-plots

clean-benchmarks: clean-confirm
	cd sws_benchmarks && make clean-benchmarks
	cd awfy_benchmarks && make clean-benchmarks
	cd grmtools_benchmarks && make clean-benchmarks

clean-builds: clean-confirm
	cd sws_benchmarks && make clean-builds
	cd awfy_benchmarks && make clean-builds
	cd grmtools_benchmarks && make clean-builds

clean-alloy: clean-confirm
	rm -rf $(BIN)
