export PEXECS ?= 10
export BENCHMARKS ?= som grmtools alacritty fd regex-redux binary-trees ripgrep
export EXPERIMENTS ?= gcvs premopt elision
export METRICS ?= perf mem

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

BENCHMARK_DIRS := $(addprefix $(PWD)/benchmarks/, $(BENCHMARKS))

export RESULTS_DIR = $(PWD)/results

RESULTS := $(foreach e,$(EXPERIMENTS),$(foreach b,$(BENCHMARKS),$(e)/$(b)))
PERF_RESULTS := $(foreach r,$(RESULTS),$(foreach m,$(METRICS),$(r)/$(m).csv))
PERF_RESULTS := $(addprefix $(RESULTS_DIR)/, $(PERF_RESULTS))

RESULTS := $(addprefix $(RESULTS_DIR)/, $(addsuffix /data.csv, $(RESULTS)))
export ALLOY_PATH = $(ALLOY_SRC)/bin
export LIBGC_PATH = $(LIBGC_SRC)/lib
export REBENCH_EXEC = $(VENV)/bin/rebench
export LD_LIBRARY_PATH = $(LIBGC_PATH)
export PLOTS_DIR = $(PWD)/plots
export REBENCH_PROCESSOR = $(PYTHON_EXEC) $(PWD)/process.py
ALLOY_TARGETS := $(addprefix $(ALLOY_PATH)/, $(ALLOY_DEFAULTS) $(ALLOY_CFGS))

XVFB_DISPLAY := :99
XVFB_CMD := Xvfb $(XVFB_DISPLAY) -screen 0 1600x1200x16 -nolisten tcp

all: build bench

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


# Start Xvfb
start-xvfb:
	@echo "Starting Xvfb on display $(XVFB_DISPLAY)..."
	$(XVFB_CMD) &
	@sleep 2
	@echo "Xvfb started."

build: build-alloy
	$(foreach b, $(BENCHMARK_DIRS), cd $(b)/ && make build;)

# bench: venv $(RESULTS)
bench: start-xvfb $(PERF_RESULTS)
	stop-xvfb

stop-xvfb:
	@echo "Stopping Xvfb on display $(XVFB_DISPLAY)..."
	@if [ -f /tmp/.X$(subst :,,$(XVFB_DISPLAY))-lock ]; then \
		echo "Killing Xvfb process..."; \
		kill $$(ps aux | grep '[X]vfb $(XVFB_DISPLAY)' | awk '{print $$2}') || true; \
		rm -f /tmp/.X$(subst :,,$(XVFB_DISPLAY))-lock; \
	else \
		echo "No Xvfb lock file found. Skipping."; \
	fi

$(RESULTS_DIR)/%.csv:
	mkdir -p $(basename $@)/metrics/runtime
	mkdir -p $(dir $@)heaptrack
	- $(REBENCH_EXEC) -R -D \
		--invocations $(PEXECS) \
		--iterations 1 \
		-df $@ $(PWD)/rebench_$(notdir $*).conf $(subst /,-,$(patsubst %/,%,$(dir $*)))

# $(RESULTS_DIR)/%/data.csv:
# 	@echo $*
# 	mkdir -p $(dir $@)metrics/runtime
# 	mkdir -p $(dir $@)metrics/heaptrack
# 	mkdir -p $(dir $@)metrics/rss
# 	- $(REBENCH_EXEC) -R -D \
# 		--invocations $(PEXECS) \
# 		--iterations 1 \
# 		-df $@ $(PWD)/rebench.conf $(subst /,-,$*)
#

# $(RESULTS_DIR)/%/perf.csv:
# 	@echo $*
# 	mkdir -p $(dir $@)perf/metrics/runtime
# 	- $(REBENCH_EXEC) -R -D \
# 		--invocations $(PEXECS) \
# 		--iterations 1 \
# 		-df $@ $(PWD)/rebench_perf.conf $(subst /,-,$*)

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
