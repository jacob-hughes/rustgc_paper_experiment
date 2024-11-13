PWD != pwd
BIN = $(PWD)/bin
PATCHES  = $(PWD)/patches

# VENV
PYTHON = python3
PYTHON_EXEC = $(VENV_DIR)/bin/python
VENV_DIR = $(PWD)/venv
PIP = $(VENV_DIR)/bin/pip

# SOURCES
EXPERIMENTS = java_parse

REGEX_REPO = https://github.com/rust-lang/regex
REGEX_VERSION = bcbe40342628b15ab2543d386c745f7f0811b791
GRMTOOLS_REPO = https://github.com/softdevteam/grmtools
GRMTOOLS_VERSION = a0972be0777e599a3dbca710fb0a595c39560b69
CACTUS_REPO = https://github.com/softdevteam/cactus
CACTUS_VERSION = 8d34c207e1479cecf0b9b2f7beb1a0c22c8949ad

REBENCH_DATA = rebench.data

.PHONY: bench bench-perf bench-barriers bench-elision
.PHONY: build build-perf build-barriers build-elision
.PHONY: plot plot-setup plot-perf plot-barriers plot-elision
.PHONY: venv

all: venv build bench plot

bench: bench-perf bench-barriers bench-elision

bench-perf: source_files
	mkdir -p $(PWD)/results/perf
	- $(REBENCH_EXEC) -R -D \
		--invocations ${PEXECS} \
		--iterations ${ITERS} \
		-df results/perf/rebench.data \
		grmtools.conf perf

bench-barriers: source_files
	mkdir -p $(PWD)/results/barriers
	- $(REBENCH_EXEC) -R -D \
		--invocations ${PEXECS} \
		--iterations ${ITERS} \
		-df results/barriers/rebench.data \
		grmtools.conf barriers

bench-elision: source_files
	mkdir -p $(PWD)/results/elision
	- $(REBENCH_EXEC) -R -D \
		--invocations ${PEXECS} \
		--iterations ${ITERS} \
		-df results/elision/rebench.data \
		grmtools.conf elision

build: build-perf build-barriers build-elision

build-perf: grmtools cactus regex
	cd $(PWD)/cactus && git reset --hard
	cd $(PWD)/parserbench && \
		RUSTC="../../bin/alloy/alloy/bin/rustc" \
		cargo build --release --target-dir=$(PWD)/arc
	cd $(PWD)/regex && git reset --hard
	cd $(PWD)/cactus && git reset --hard
	cd $(PWD)/regex && git apply $(PWD)/regex_gc.patch
	cd $(PWD)/cactus && git apply $(PWD)/cactus_gc.patch
	cd $(PWD)/parserbench && \
		RUSTC="../../bin/alloy/alloy/bin/rustc" \
		cargo build --release --target-dir=$(PWD)/gc
	cd $(PWD)/cactus && git reset --hard
	cd $(PWD)/regex && git reset --hard

build-barriers: grmtools cactus regex
	cd $(PWD)/regex && git apply $(PWD)/regex_gc.patch
	cd $(PWD)/cactus && git diff-index --quiet HEAD --
	cd $(PWD)/cactus && git apply $(PWD)/cactus_gc.patch
	cd $(PWD)/parserbench && \
		RUSTC="../../bin/alloy/barriers_naive/bin/rustc" \
		cargo build --release --target-dir=$(PWD)/barriers_naive
	cd $(PWD)/parserbench && \
		RUSTC="../../bin/alloy/barriers_none/bin/rustc" \
		cargo build --release --target-dir=$(PWD)/barriers_none
	cd $(PWD)/parserbench && \
		RUSTC="../../bin/alloy/barriers_opt/bin/rustc" \
		cargo build --release --target-dir=$(PWD)/barriers_opt
	cd $(PWD)/cactus && git reset --hard
	cd $(PWD)/regex && git reset --hard

build-elision: grmtools cactus regex
	cd $(PWD)/regex && git apply $(PWD)/regex_gc.patch
	cd $(PWD)/cactus && git diff-index --quiet HEAD --
	cd $(PWD)/cactus && git apply $(PWD)/cactus_gc.patch
	cd $(PWD)/parserbench && \
		RUSTC="../../bin/alloy/finalise_elide/bin/rustc" \
		cargo build --release --target-dir=$(PWD)/finalise_elide
	cd $(PWD)/parserbench && \
		RUSTC="../../bin/alloy/finalise_naive/bin/rustc" \
		cargo build --release --target-dir=$(PWD)/finalise_naive
	cd $(PWD)/cactus && git reset --hard
	cd $(PWD)/regex && git reset --hard

plot:
	mkdir -p plots
	$(PYTHON_EXEC) $(REBENCH_PROCESSOR) summary grmtools

regex:
	git clone $(REGEX_REPO) regex
	cd regex && git checkout $(REGEX_VERSION)

grmtools:
	git clone $(GRMTOOLS_REPO) grmtools
	cd grmtools && git checkout $(GRMTOOLS_VERSION)
	cd $(PWD)/grmtools && git apply $(PWD)/grmtools.patch
cactus:
	git clone $(CACTUS_REPO) cactus
	cd cactus && git checkout $(CACTUS_VERSION)


source_files: hadoop eclipse jenkins spring

hadoop:
	git clone https://github.com/apache/hadoop --depth 1

eclipse:
	git clone https://github.com/eclipse-platform/eclipse.platform eclipse --depth 1
spring:
	git clone https://github.com/spring-projects/spring-framework --depth 1 spring

jenkins:
	git clone https://github.com/jenkinsci/jenkins jenkins --depth 1

clean-benchmarks:
	rm -rf results

clean-builds:
	rm -rf $(PWD)/bin
	rm -rf $(PWD)/cactus
	rm -rf $(PWD)/grmtools
	rm -rf $(PWD)/regex
	rm -rf $(PWD)/arc
	rm -rf $(PWD)/gc
	rm -rf $(PWD)/barriers_naive
	rm -rf $(PWD)/barriers_none
	rm -rf $(PWD)/barriers_opt
	rm -rf $(PWD)/finalise_elide
	rm -rf $(PWD)/finalise_naive
	rm -rf jenkins spring eclipse hadoop

clean-plots:
	rm -rf plots

clean-confirm:
	@( read -p "Are you sure? [y/N]: " sure && case "$$sure" in [yY]) true;; *) false;; esac )

clean: clean-confirm clean-builds clean-benchmarks clean-plots
	@echo "Clean"

venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: ../requirements.txt
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install -r ../requirements.txt
