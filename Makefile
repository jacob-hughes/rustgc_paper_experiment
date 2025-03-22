export PEXECS ?= 10

PWD != pwd

PYTHON = python3
VENV = $(PWD)/venv
PIP = $(VENV)/bin/pip
PYTHON_EXEC = $(VENV)/bin/python

ALLOY_REPO = https://github.com/jacob-hughes/alloy
ALLOY_VERSION = fix_stats
ALLOY_BOOTSTRAP_STAGE = 1
ALLOY_SRC = $(PWD)/alloy

CFGS = $(subst .,/,$(notdir $(patsubst %.config.toml,%,$(wildcard $(PWD)/configs/*))))
export ALLOY_DEFAULTS := $(addprefix gcvs/, perf mem)
export ALLOY_CFGS := $(filter-out $(ALLOY_DEFAULTS), $(CFGS))

LIBGC_REPO = https://github.com/jacob-hughes/bdwgc
LIBGC_VERSION = gc_reclaimed
LIBGC_SRC = $(PWD)/bdwgc

HEAPTRACK_REPO = https://github.com/kde/heaptrack
HEAPTRACK_VERSION = master
HEAPTRACK_SRC = $(PWD)/heaptrack
HEAPTRACK = $(HEAPTRACK_SRC)/bin

BENCHMARKS = grmtools ripgrep alacritty fd som
# BENCHMARKS = som grmtools alacritty fd regex-redux binary-trees ripgrep
# BENCHMARKS = regex-redux
# BENCHMARKS = som grmtools binary-trees grmtools
BENCHMARK_DIRS := $(addprefix $(PWD)/benchmarks/, $(BENCHMARKS))

export RESULTS_DIR = $(PWD)/results

export EXPERIMENTS = gcvs premopt elision
RESULTS := $(foreach e,$(EXPERIMENTS),$(foreach b,$(BENCHMARKS),$(e)/$(b)))
RESULTS := $(addprefix $(RESULTS_DIR)/, $(addsuffix /data.csv, $(RESULTS)))

export ALLOY_PATH = $(ALLOY_SRC)/bin
export LIBGC_PATH = $(LIBGC_SRC)/lib
export REBENCH_EXEC = $(VENV)/bin/rebench
export LD_LIBRARY_PATH = $(LIBGC_PATH)
export PLOTS_DIR = $(PWD)/plots
export REBENCH_PROCESSOR = $(PYTHON_EXEC) $(PWD)/process.py
ALLOY_TARGETS := $(addprefix $(ALLOY_PATH)/, $(ALLOY_DEFAULTS) $(ALLOY_CFGS))

all: build

.PHONY: venv
.PHONY: build build-alloy
.PHONY: bench plot
.PHONY: clean clean-alloy clean-results clean-plots clean-confirm

build-alloy: $(ALLOY_SRC) $(LIBGC_PATH) $(HEAPTRACK) $(ALLOY_TARGETS)

$(ALLOY_PATH)/%:
	@echo $@
	RUSTFLAGS="-L $(LIBGC_SRC)/lib" \
	$(ALLOY_SRC)/x install \
		--config $(PWD)/configs/$(subst /,.,$*).config.toml \
		--stage $(ALLOY_BOOTSTRAP_STAGE) \
		--set build.docs=false \
		--set install.prefix=$(ALLOY_SRC)/bin/$* \
		--set install.sysconfdir=etc

$(ALLOY_SRC):
	git clone $(ALLOY_REPO) $@
	cd $@ && git checkout $(ALLOY_VERSION)

$(LIBGC_SRC):
	git clone $(LIBGC_REPO) $@
	cd $@ && git checkout $(LIBGC_VERSION)

$(LIBGC_PATH): $(LIBGC_SRC)
	mkdir -p $</build
	cd $</build && cmake -DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_INSTALL_PREFIX="$(LIBGC_SRC)" \
		-DCMAKE_C_FLAGS="-DGC_ALWAYS_MULTITHREADED -DVALGRIND_TRACKING" ../ && \
		make -j$(numproc) install

$(HEAPTRACK_SRC):
	git clone $(HEAPTRACK_REPO) $@
	cd $@ && git checkout $(HEAPTRACK_VERSION)

$(HEAPTRACK): $(HEAPTRACK_SRC)
	mkdir -p $</build
	cd $</build && \
		cmake -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_INSTALL_PREFIX=$(HEAPTRACK_SRC) ../ && \
		make -j$(numproc) install


build: build-alloy
	$(foreach b, $(BENCHMARK_DIRS), cd $(b)/ && make build;)

bench: venv $(RESULTS)

$(RESULTS_DIR)/%/data.csv:
	@echo $*
	mkdir -p $(dir $@)metrics/runtime
	mkdir -p $(dir $@)metrics/heaptrack
	mkdir -p $(dir $@)metrics/rss
	- $(REBENCH_EXEC) -R -D \
		--invocations $(PEXECS) \
		--iterations 1 \
		-df $@ $(PWD)/rebench.conf $(subst /,-,$*)

plot: venv
	$(REBENCH_PROCESSOR) $(PLOTS_DIR) $(RESULTS_DIR)

venv: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt

clean: clean-confirm clean-alloy clean-builds
	rm -rf $(ALLOY_SRC) $(HEAPTRACK_SRC) $(LIBGC_SRC)
	@echo "Clean"

clean-alloy:
	rm -rf $(ALLOY_TARGETS)

clean-builds:
	$(foreach b, $(BENCHMARK_DIRS), cd $(b)/ && make clean-builds;)

clean-confirm:
	@echo $@
	rm -rf $(RESULTS_DIR) $(PLOTS_DIR)
	@( read -p "Are you sure? [y/N]: " sure && case "$$sure" in [yY]) true;; *) false;; esac )
