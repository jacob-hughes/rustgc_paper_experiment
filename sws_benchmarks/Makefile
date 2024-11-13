TODAY ?= $(shell date -u '+%B %Y')
PWD != pwd

# VENV
PYTHON = python3
PYTHON_EXEC = $(VENV_DIR)/bin/python
VENV_DIR = $(PWD)/venv
PIP = $(VENV_DIR)/bin/pip

# SOURCES
SWS_REPO = https://github.com/static-web-server/static-web-server
SWS_VERSION = 30a640961322d7f8aa2ffa0ab6f409134ed97eea
WRK_REPO = https://github.com/wg/wrk
WRK_VERSION = a211dd5a7050b1f9e8a9870b95513060e72ac4a0
HYPER_REPO = https://github.com/hyperium/hyper
HYPER_VERSION = f8e2a831943e9a687a1b3277c806caca6e0ac8a4
PATCHES = $(PWD)/patches

# BENCHMARK RUNNER
PEXECS ?= 2
BARRIERS = barriers_naive barriers_none barriers_opt
ELISION = finalise_elide finalise_naive
PERF = arc gc
URL ?= "http://localhost:8787"
THREADS ?= 4
CONNECTIONS ?= 100
DURATION ?= 3s
WRK = $(PWD)/wrk/wrk

make all: venv build bench plot

.PHONY: $(PERF) $(BARRIERS) $(ELISION)
.PHONY: bench bench-perf bench-barriers bench-elision
.PHONY: build build-perf build-barriers build-elision
.PHONY: plot plot-setup plot-perf plot-barriers plot-elision
.PHONY: venv

bench: bench-perf bench-barriers bench-elision

bench-perf: $(PERF)

bench-barriers: $(BARRIERS)

bench-elision: $(ELISION)

build: build-perf build-barriers build-elision

plot: plot-perf plot-barriers plot-elision

plot-perf: plot-setup
	@echo "TODO: plot-perf"

plot-barriers: plot-setup
	@echo "TODO: plot-barriers"

plot-elision: plot-setup
	@echo "TODO: plot-elision"

plot-setup: venv
	mkdir -p plots
	$(PYTHON_EXEC) merge_results.py

$(PERF):
	@if [ -f "results/raw_$@.csv" ]; then \
		rm -f "results/raw_$@.csv"; \
	fi
	mkdir -p $@/www
	PEXECS=$(PEXECS) SERVER=$@ ./run.sh $(PWD)/$@ $(WRK) $(CONNECTIONS) $(THREADS) $(DURATION) $(URL)

$(BARRIERS):
	@if [ -f "results/raw_$@.csv" ]; then \
		rm -f "results/raw_$@.csv"; \
	fi
	mkdir -p $@/www
	PEXECS=$(PEXECS) SERVER=$@ ./run.sh $(PWD)/$@ $(WRK) $(CONNECTIONS) $(THREADS) $(DURATION) $(URL)

$(ELISION):
	@if [ -f "results/raw_$@.csv" ]; then \
		rm -f "results/raw_$@.csv"; \
	fi
	mkdir -p $@/www
	PEXECS=$(PEXECS) SERVER=$@ ./run.sh $(PWD)/$@ $(WRK) $(CONNECTIONS) $(THREADS) $(DURATION) $(URL)

build-perf: sws hyper wrk
	cd sws && git reset --hard
	cd hyper && git reset --hard
	cd sws && git apply $(PATCHES)/arc/sws.patch
	cd sws && \
		RUSTC="../../bin/alloy/alloy/bin/rustc" \
		cargo build --release --target-dir=$(PWD)/arc
	cd sws && git reset --hard
	cd sws && git apply $(PATCHES)/gc/sws.patch
	cd hyper && git reset --hard
	cd hyper && git apply $(PATCHES)/gc/hyper.patch
	cd sws && \
		RUSTC="../../bin/alloy/alloy/bin/rustc" \
		cargo build --release --target-dir=$(PWD)/gc
	cd sws && git reset --hard

build-barriers: sws hyper wrk
	cd hyper && git reset --hard
	cd hyper && git apply $(PATCHES)/gc/hyper.patch
	cd sws && git reset --hard
	cd sws && git apply $(PATCHES)/gc/sws.patch
	cd sws && \
		RUSTC="../../bin/alloy/barriers_naive/bin/rustc" \
		cargo build --release --target-dir=$(PWD)/barriers_naive
	cd sws && \
		RUSTC="../../bin/alloy/barriers_none/bin/rustc" \
		cargo build --release --target-dir=$(PWD)/barriers_none
	cd sws && \
		RUSTC="../../bin/alloy/barriers_opt/bin/rustc" \
		cargo build --release --target-dir=$(PWD)/barriers_opt
	cd sws && git reset --hard
	cd hyper && git reset --hard

build-elision: sws hyper wrk
	cd hyper && git reset --hard
	cd hyper && git apply $(PATCHES)/gc/hyper.patch
	cd sws && git reset --hard
	cd sws && git apply $(PATCHES)/gc/sws.patch
	cd sws && \
		RUSTC="../../bin/alloy/finalise_elide/bin/rustc" \
		cargo build --release --target-dir=$(PWD)/finalise_elide
	cd sws && \
		RUSTC="../../bin/alloy/finalise_naive/bin/rustc" \
		cargo build --release --target-dir=$(PWD)/finalise_naive
	cd sws && git reset --hard
	cd hyper && git reset --hard

sws:
	git clone $(SWS_REPO) sws
	cd sws && git checkout $(SWS_VERSION)

hyper:
	git clone $(HYPER_REPO) hyper
	cd hyper && git checkout $(HYPER_VERSION)

wrk:
	git clone $(WRK_REPO) wrk
	cd wrk && git checkout $(WRK_VERSION)
	cd wrk && make -j$(nprocs)

venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: ../requirements.txt
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install -r ../requirements.txt

clean-builds:
	rm -rf arc
	rm -rf gc
	rm -rf barriers_opt
	rm -rf barriers_naive
	rm -rf barriers_none
	rm -rf finalise_naive
	rm -rf finalise_elide

clean-src:
	rm -rf sws
	rm -rf hyper
	rm -rf wrk

clean-benchmarks:
	rm -rf results

clean-plots:
	rm -rf plots

clean-confirm:
	@( read -p "Are you sure? [y/N]: " sure && case "$$sure" in [yY]) true;; *) false;; esac )

clean: clean-confirm clean-src clean-builds clean-benchmarks clean-plots
	@echo "Clean"
