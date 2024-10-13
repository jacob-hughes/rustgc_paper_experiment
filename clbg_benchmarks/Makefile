PWD != pwd
PYTHON = python3
BIN = $(PWD)/bin

EXPERIMENTS = binary_trees regex_redux
REGEX_VERSION = bcbe40342628b15ab2543d386c745f7f0811b791
REGEX_REPO = https://github.com/rust-lang/regex

.PHONY: build
.PHONY: clean clean-builds
.PHONY: $(EXPERIMENTS)

all: $(EXPERIMENTS)

build: $(BIN)/regex_redux  $(BIN)/binary_trees

plot:
	$(PYTHON_EXEC) $(REBENCH_PROCESSOR) summary binary_trees regex_redux

clean-plots:
	- rm summary.csv

$(EXPERIMENTS): build
	mkdir -p $(PWD)/results/$@
	- $(REBENCH_EXEC) -R -D \
		--invocations ${PEXECS} \
		--iterations ${ITERS} \
		-df results/$(REBENCH_DATA) \
		clbg.conf $@

$(BIN)/regex_redux: regex-rc regex-gc
	mkdir -p $@
	cd $(PWD)/regex_redux && ln -s Cargo.toml.gc Cargo.toml
	cd $(PWD)/regex_redux && cargo build --release --bin alloy
	cd $(PWD)/regex_redux && rm Cargo.toml && ln -s Cargo.toml.rc Cargo.toml
	cd $(PWD)/regex_redux && cargo build --release --bin rc
	cd $(PWD)/regex_redux && cargo build --release --bin fasta
	$(PWD)/regex_redux/target/release/fasta 2500000 > redux_input.txt
	cd $(PWD)/regex_redux && rm Cargo.toml
	@echo "Built with RUSTC=$$RUSTC"
	ln -s $(PWD)/$(notdir $@)/target/release/* $@/

$(BIN)/binary_trees:
	mkdir -p $@
	cd $(PWD)/binary_trees && cargo build --release
	@echo "Built with RUSTC=$$RUSTC"
	ln -s $(PWD)/$(notdir $@)/target/release/* $@/

regex/alloy: regex
	cd regex && git diff-index --quiet HEAD --
	cd regex && git apply $(PWD)/regex_gc.patch
	cd regex && cargo build --release --target-dir=alloy
	cd regex && git reset --hard


regex-gc: regex-rc
	cp -r regex-rc regex-gc
	cd regex-gc && git apply $(PWD)/regex-gc.patch

regex-rc:
	git clone $(REGEX_REPO) regex-rc
	cd regex-rc && git checkout $(REGEX_VERSION)

clean-benchmarks:
	rm -rf results

clean-builds:
	rm -rf $(PWD)/binary_trees/target
	rm -rf $(PWD)/regex_redux/target
	rm -rf $(PWD)/regex-rc
	rm -rf $(PWD)/regex-gc
	rm -rf $(PWD)/bin
	rm -rf redux_input.txt

clean: clean-builds clean-benchmarks
