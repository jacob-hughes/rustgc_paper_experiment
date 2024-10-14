PWD != pwd
BIN = $(PWD)/bin

EXPERIMENTS = java_parse

REGEX_REPO = https://github.com/rust-lang/regex
GRMTOOLS_REPO = https://github.com/softdevteam/grmtools
CACTUS_REPO = https://github.com/softdevteam/cactus

REGEX_VERSION = bcbe40342628b15ab2543d386c745f7f0811b791
GRMTOOLS_VERSION = a0972be0777e599a3dbca710fb0a595c39560b69
CACTUS_VERSION = 8d34c207e1479cecf0b9b2f7beb1a0c22c8949ad

.PHONY: build
.PHONY: clean clean-builds

all: bench

bench: source_files
	mkdir -p $(PWD)/results/grmtools
	- $(REBENCH_EXEC) -R -D \
		--invocations ${PEXECS} \
		--iterations ${ITERS} \
		-df results/grmtools/$(REBENCH_DATA) \
		grmtools.conf perf

build: grmtools cactus regex $(BIN)/gc $(BIN)/rc

plot:
	mkdir -p plots
	$(PYTHON_EXEC) $(REBENCH_PROCESSOR) summary grmtools

clean-plots:
	- rm summary.csv

$(BIN)/gc:
	mkdir -p $@
	cd $(PWD)/regex && git apply $(PWD)/regex_gc.patch
	cd $(PWD)/cactus && git diff-index --quiet HEAD --
	cd $(PWD)/cactus && git apply $(PWD)/cactus_gc.patch
	cd $(PWD)/parserbench && cargo build --release --target-dir=$(BIN)/gc
	cd $(PWD)/cactus && git reset --hard
	cd $(PWD)/regex && git reset --hard

$(BIN)/rc:
	mkdir -p $@
	cd $(PWD)/cactus && git diff-index --quiet HEAD --
	cd $(PWD)/parserbench && cargo build --release --target-dir=$(BIN)/rc

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
	cd hadoop && git checkout e4b070025b38d8866dd1e2dc4a3f46b8d9db4ad5

eclipse:
	git clone https://github.com/eclipse-platform/eclipse.platform eclipse --depth 1
	cd eclipse && git checkout e14565e5022852f2e1eebadbe12d09f4cee88378
spring:
	git clone https://github.com/spring-projects/spring-framework --depth 1 spring
	cd spring && git checkout 60978ff4c7fdcda9406ddad6b04bd2f1237fb77e

jenkins:
	git clone https://github.com/jenkinsci/jenkins jenkins --depth 1
	cd jenkins && git checkout 01ee3ad4d4c5c455c9ae48f3cde947c7f8f80f02

# elasticsearch:
# 	git clone https://github.com/elastic/elasticsearch --depth 1
# 	cd elasticsearch && git checkout 74522c43d80c7eaf67b12eccb841fef74b85f808

clean-benchmarks:
	rm -rf results

clean-builds:
	rm -rf $(PWD)/bin
	rm -rf $(PWD)/cactus
	rm -rf $(PWD)/grmtools
	rm -rf $(PWD)/regex

clean: clean-builds clean-benchmarks

venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install -r requirements.txt

