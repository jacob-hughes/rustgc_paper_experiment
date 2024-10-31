PWD != pwd
BIN = $(PWD)/bin
PATCH_DIR  = $(PWD)/patches

SOMRS_REPO = https://github.com/Hirevo/som-rs
YKSOM_REPO = https://github.com/softdevteam/yksom

SOMRS_VERSION = 35b780cbee765cca24201fe063d3f1055ec7f608
YKSOM_VERSION=fc7c7c131ba93b7e3c85a172fbcc245f29c324d6

YKSOM_EXPS = $(addprefix results/, yksom_elision yksom_barriers)

SOMRSBC_EXPS = $(addprefix results/, som_rs_bc_elision som_rs_bc_barriers som_rs_bc_perf)
SOMRSAST_EXPS = $(addprefix results/, som_rs_ast_elision som_rs_ast_barriers som_rs_ast_perf)

EXPERIMENTS = $(SOMRSAST_EXPS) $(SOMRSBC_EXPS) $(YKSOM_EXPS)

all: bench

.PHONY: build bench
.PHONY: clean clean-benchmarks clean-builds
.PHONY: $(EXPERIMENTS)

plot:
	mkdir -p plots
	# $(PYTHON_EXEC) $(REBENCH_PROCESSOR) summary som_rs_bc_perf som_rs_ast_perf
	$(PYTHON_EXEC) $(REBENCH_PROCESSOR) elision som_rs_bc_elision som_rs_ast_elision
	$(PYTHON_EXEC) $(REBENCH_PROCESSOR) barriers som_rs_bc_barriers som_rs_ast_barriers
	# $(PYTHON_EXEC) $(REBENCH_PROCESSOR) barriers som_rs_ast_barriers

clean-plots:
	rm -rf plots
	- rm summary.csv

bench:  $(EXPERIMENTS)

$(YKSOM_EXPS):
	mkdir -p $@
	- $(REBENCH_EXEC) -R -D \
		--invocations ${PEXECS} \
		--iterations ${ITERS} \
		-df $@/$(REBENCH_DATA) \
		yksom.conf $(notdir $@)

$(SOMRSAST_EXPS):
	mkdir -p $@
	- $(REBENCH_EXEC) -R -D \
		--invocations ${PEXECS} \
		--iterations ${ITERS} \
		-df $@/$(REBENCH_DATA) \
		som-rs-ast.conf $(notdir $@)

$(SOMRSBC_EXPS):
	mkdir -p $@
	- $(REBENCH_EXEC) -R -D \
		--invocations ${PEXECS} \
		--iterations ${ITERS} \
		-df $@/$(REBENCH_DATA) \
		som-rs-bc.conf $(notdir $@)

build: build-som-rs build-yksom

build-som-rs: som-rs/Cargo.lock som-rs-finalise-naive som-rs-finalise-elide som-rs-barriers

build-yksom: yksom/Cargo.toml build-yksom-finalise-naive build-yksom-finalise-elide \
	build-yksom-barriers-naive build-yksom-barriers-none build-yksom-barriers-opt

som-rs-barriers:
	mkdir -p $(BIN)/som-rs/{barriers_naive,barriers_none,barriers_opt}
	cd $(PWD)/som-rs && git reset --hard
	cd som-rs && git apply $(PATCH_DIR)/som-rs/use_gc.patch
	cd som-rs && git apply $(PATCH_DIR)/som-rs/dump_stats.patch
	cd som-rs && \
		RUSTC="../../bin/alloy/barriers_naive/bin/rustc" \
		cargo build --release --target-dir=barriers_naive
	cd som-rs && \
		RUSTC="../../bin/alloy/barriers_none/bin/rustc" \
		cargo build --release --target-dir=barriers_none
	cd som-rs && \
		RUSTC="../../bin/alloy/barriers_opt/bin/rustc" \
		cargo build --release --target-dir=barriers_opt
	ln -s $(PWD)/som-rs/core-lib $(BIN)/som-rs/core-lib
	ln -s $(PWD)/som-rs/barriers_naive/release/som-interpreter-ast $(BIN)/som-rs/barriers_naive/som-rs-ast
	ln -s $(PWD)/som-rs/barriers_none/release/som-interpreter-ast $(BIN)/som-rs/barriers_none/som-rs-ast
	ln -s $(PWD)/som-rs/barriers_opt/release/som-interpreter-ast $(BIN)/som-rs/barriers_opt/som-rs-ast
	ln -s $(PWD)/som-rs/barriers_naive/release/som-interpreter-bc $(BIN)/som-rs/barriers_naive/som-rs-bc
	ln -s $(PWD)/som-rs/barriers_none/release/som-interpreter-bc $(BIN)/som-rs/barriers_none/som-rs-bc
	ln -s $(PWD)/som-rs/barriers_opt/release/som-interpreter-bc $(BIN)/som-rs/barriers_opt/som-rs-bc

som-rs-finalise-elide:
	- rm -rf $(BIN)/som-rs/finalise_elide
	mkdir -p $(BIN)/som-rs/finalise_elide
	cd som-rs && git reset --hard
	cd som-rs && git apply $(PATCH_DIR)/som-rs/use_gc.patch
	cd som-rs && git apply $(PATCH_DIR)/som-rs/dump_stats.patch
	cd som-rs && \
		RUSTC="../../bin/alloy/finalise_elide/bin/rustc" \
		cargo build --release --target-dir=finalise_elide
	ln -s $(PWD)/som-rs/core-lib $(BIN)/som-rs/finalise_elide/core-lib
	ln -s $(PWD)/som-rs/finalise_elide/release/som-interpreter-ast $(BIN)/som-rs/finalise_elide/som-rs-ast
	ln -s $(PWD)/som-rs/finalise_elide/release/som-interpreter-bc $(BIN)/som-rs/finalise_elide/som-rs-bc

som-rs-finalise-naive:
	- rm -rf $(BIN)/som-rs/finalise_naive
	mkdir -p $(BIN)/som-rs/finalise_naive
	cd som-rs && git reset --hard
	cd som-rs && git apply $(PATCH_DIR)/som-rs/use_gc.patch
	cd som-rs && git apply $(PATCH_DIR)/som-rs/dump_stats.patch
	cd som-rs && \
		RUSTC="../../bin/alloy/finalise_naive/bin/rustc" \
		cargo build --release --target-dir=finalise_naive
	ln -s $(PWD)/som-rs/core-lib $(BIN)/som-rs/finalise_naive/core-lib
	ln -s $(PWD)/som-rs/finalise_naive/release/som-interpreter-ast $(BIN)/som-rs/finalise_naive/som-rs-ast
	ln -s $(PWD)/som-rs/finalise_naive/release/som-interpreter-bc $(BIN)/som-rs/finalise_naive/som-rs-bc

som-rs-perf-rc:
	- rm -rf $(BIN)/som-rs/perf_rc
	mkdir -p $(BIN)/som-rs/perf_rc
	cd som-rs && git reset --hard
	cd som-rs && git apply $(PATCH_DIR)/som-rs/bdwgc_allocator.patch
	cd som-rs && git apply $(PATCH_DIR)/som-rs/dump_stats.patch
	cd som-rs && \
		RUSTC="../../bin/alloy/alloy/bin/rustc" \
		cargo build --release --target-dir=perf_rc
	ln -s $(PWD)/som-rs/core-lib $(BIN)/som-rs/perf_rc/core-lib
	ln -s $(PWD)/som-rs/perf_rc/release/som-interpreter-ast $(BIN)/som-rs/perf_rc/som-rs-ast
	ln -s $(PWD)/som-rs/perf_rc/release/som-interpreter-bc $(BIN)/som-rs/perf_rc/som-rs-bc

som-rs-perf-gc:
	- rm -rf $(BIN)/som-rs/perf_gc
	mkdir -p $(BIN)/som-rs/perf_gc
	cd som-rs && git reset --hard
	cd som-rs && git apply $(PATCH_DIR)/som-rs/use_gc.patch
	cd som-rs && git apply $(PATCH_DIR)/som-rs/dump_stats.patch
	cd som-rs && \
		RUSTC="../../bin/alloy/alloy/bin/rustc" \
		cargo build --release --target-dir=perf_gc
	ln -s $(PWD)/som-rs/core-lib $(BIN)/som-rs/perf_gc/core-lib
	ln -s $(PWD)/som-rs/perf_gc/release/som-interpreter-ast $(BIN)/som-rs/perf_gc/som-rs-ast
	ln -s $(PWD)/som-rs/perf_gc/release/som-interpreter-bc $(BIN)/som-rs/perf_gc/som-rs-bc

build-yksom-finalise-naive:
	- rm -rf $(BIN)/yksom/finalise_naive
	mkdir -p $(BIN)/yksom/finalise_naive
	cd yksom && git reset --hard
	cd yksom && git apply $(PATCH_DIR)/yksom/dump_stats.patch
	cd yksom && RUSTC="../../bin/alloy/finalise_naive/bin/rustc" \
	      cargo build --release --target-dir=finalise_naive
	ln -s $(PWD)/yksom/SOM $(BIN)/yksom/finalise_naive/SOM
	ln -s $(PWD)/yksom/finalise_naive/release/yksom $(BIN)/yksom/finalise_naive/yksom

build-yksom-finalise-elide:
	- rm -rf $(BIN)/yksom/finalise_elide
	mkdir -p $(BIN)/yksom/finalise_elide
	cd yksom && git reset --hard
	cd yksom && git apply $(PATCH_DIR)/yksom/dump_stats.patch
	cd yksom && RUSTC="../../bin/alloy/finalise_elide/bin/rustc" \
	      cargo build --release --target-dir=finalise_elide
	ln -s $(PWD)/yksom/SOM $(BIN)/yksom/finalise_elide/SOM
	ln -s $(PWD)/yksom/finalise_elide/release/yksom $(BIN)/yksom/finalise_elide/yksom

build-yksom-barriers-naive:
	- rm -rf $(BIN)/yksom/barriers_naive
	mkdir -p $(BIN)/yksom/barriers_naive
	cd yksom && git reset --hard
	cd yksom && git apply $(PATCH_DIR)/yksom/dump_stats.patch
	cd yksom && RUSTC="../../bin/alloy/barriers_naive/bin/rustc" \
	      cargo build --release --target-dir=barriers_naive
	ln -s $(PWD)/yksom/SOM $(BIN)/yksom/barriers_naive/SOM
	ln -s $(PWD)/yksom/barriers_naive/release/yksom $(BIN)/yksom/barriers_naive/yksom

build-yksom-barriers-opt:
	- rm -rf $(BIN)/yksom/barriers_opt
	mkdir -p $(BIN)/yksom/barriers_opt
	cd yksom && git reset --hard
	cd yksom && git apply $(PATCH_DIR)/yksom/dump_stats.patch
	cd yksom && RUSTC="../../bin/alloy/barriers_opt/bin/rustc" \
	      cargo build --release --target-dir=barriers_opt
	ln -s $(PWD)/yksom/SOM $(BIN)/yksom/barriers_opt/SOM
	ln -s $(PWD)/yksom/barriers_opt/release/yksom $(BIN)/yksom/barriers_opt/yksom

build-yksom-barriers-none:
	- rm -rf $(BIN)/yksom/barriers_none
	mkdir -p $(BIN)/yksom/barriers_none
	cd yksom && git reset --hard
	cd yksom && git apply $(PATCH_DIR)/yksom/dump_stats.patch
	cd yksom && RUSTC="../../bin/alloy/barriers_none/bin/rustc" \
	      cargo build --release --target-dir=barriers_none
	ln -s $(PWD)/yksom/SOM $(BIN)/yksom/barriers_none/SOM
	ln -s $(PWD)/yksom/barriers_none/release/yksom $(BIN)/yksom/barriers_none/yksom

clean-builds:
	rm -rf som-rs
	rm -rf yksom
	rm -rf bin

som-rs/Cargo.lock:
	git clone --recursive $(SOMRS_REPO) som-rs
	cd som-rs && git checkout $(SOMRS_VERSION)

yksom/Cargo.toml:
	git clone --recursive $(YKSOM_REPO) yksom
	cd yksom && git checkout $(YKSOM_VERSION)

clean-benchmarks:
	rm -rf results

clean: clean-benchmarks clean-builds
