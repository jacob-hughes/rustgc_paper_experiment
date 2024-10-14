PWD != pwd
BIN = $(PWD)/bin
PATCH_DIR  = $(PWD)/patches

SOMRS_REPO = https://github.com/Hirevo/som-rs
YKSOM_REPO = https://github.com/softdevteam/yksom

SOMRS_VERSION = 35b780cbee765cca24201fe063d3f1055ec7f608
YKSOM_VERSION=fc7c7c131ba93b7e3c85a172fbcc245f29c324d6

SOMRS_CFGS = $(addprefix $(BIN)/som-rs/, finalise_elide finalise_naive \
	     barriers_naive barriers_opt barriers_none \
	     perf_rc perf_gc)
YKSOM_CFGS = $(addprefix $(BIN)/yksom/, finalise_elide finalise_naive barriers_naive \
	     barriers_opt barriers_none)

SOMRSBC_EXPS = $(addprefix results/, som_rs_bc_elision som_rs_bc_perf som_rs_bc_barriers)
SOMRSAST_EXPS = $(addprefix results/, som_rs_ast_elision som_rs_ast_perf som_rs_ast_barriers)
YKSOM_EXPS = $(addprefix results/, yksom_elision yksom_barriers)
EXPERIMENTS = $(SOMRSBC_EXPS) $(SOMRSAST_EXPS) $(YKSOM_EXPS)

all: bench

.PHONY: build bench
.PHONY: clean clean-benchmarks clean-builds

plot:
	# $(PYTHON_EXEC) $(REBENCH_PROCESSOR) summary som_rs_perf
	$(PYTHON_EXEC) $(REBENCH_PROCESSOR) elision yksom_elision som_rs_bc_elision som_rs_ast_elision
	$(PYTHON_EXEC) $(REBENCH_PROCESSOR) barriers yksom_barriers som_rs_bc_barriers som_rs_ast_barriers

clean-plots:
	- rm summary.csv

bench: $(EXPERIMENTS)

$(YKSOM_EXPS): build
	mkdir -p $@
	- $(REBENCH_EXEC) -R -D \
		--invocations ${PEXECS} \
		--iterations ${ITERS} \
		-df $@/$(REBENCH_DATA) \
		yksom.conf $(notdir $@)

$(SOMRSAST_EXPS): build
	mkdir -p $@
	- $(REBENCH_EXEC) -R -D \
		--invocations ${PEXECS} \
		--iterations ${ITERS} \
		-df $@/$(REBENCH_DATA) \
		som-rs-ast.conf $(notdir $@)

$(SOMRSBC_EXPS): build
	mkdir -p $@
	- $(REBENCH_EXEC) -R -D \
		--invocations ${PEXECS} \
		--iterations ${ITERS} \
		-df $@/$(REBENCH_DATA) \
		som-rs-bc.conf $(notdir $@)

build: $(YKSOM_CFGS) $(SOMRS_CFGS)

$(YKSOM_CFGS): yksom/Cargo.toml
	cd yksom && git diff-index --quiet HEAD --
	cd yksom && git apply $(PATCH_DIR)/yksom/dump_stats.patch
	@if [ -n "$(filter $(ALLOY_CFGS), $(notdir $@))" ]; then \
		cd yksom && \
		RUSTC="../../bin/alloy/$(notdir $@)/bin/rustc" \
			cargo build --release --target-dir=$@; \
	else \
		cd yksom && \
		RUSTC="../../bin/alloy/$(ALLOY_DEFAULT_CFG)/bin/rustc" \
			cargo build --release --target-dir=$@; \
	fi
	ln -s $(PWD)/yksom/SOM $@/SOM
	ln -s $@/release/yksom $@/yksom
	cd yksom && git reset --hard

$(SOMRS_CFGS): som-rs/Cargo.lock
	cd som-rs && git diff-index --quiet HEAD --
	@if [  "$(notdir $@))" = "perf_rc" ]; then \
		cd som-rs && git apply $(PATCH_DIR)/som-rs/bdwgc_allocator.patch; \
	else \
		cd som-rs; \
		git apply $(PATCH_DIR)/som-rs/use_gc.patch; \
		echo "Applying dump stats";\
		git apply $(PATCH_DIR)/som-rs/dump_stats.patch; \
	fi
	@if [ -n "$(filter $(ALLOY_CFGS), $(notdir $@))" ]; then \
		cd som-rs && \
		RUSTC="../../bin/alloy/$(notdir $@)/bin/rustc" \
			cargo build \
				--release \
			      	--target-dir=$@; \
	else \
		cd som-rs && \
		RUSTC="../../bin/alloy/$(ALLOY_DEFAULT_CFG)/bin/rustc" \
			cargo build \
				--release \
			      	--target-dir=$@; \
	fi
	ln -s $(PWD)/som-rs/core-lib $@/core-lib
	ln -s $@/release/som-interpreter-bc $@/som-rs-bc
	ln -s $@/release/som-interpreter-ast $@/som-rs-ast
	cd som-rs && git reset --hard

som-rs/Cargo.lock:
	git clone --recursive $(SOMRS_REPO) som-rs
	cd som-rs && git checkout $(SOMRS_VERSION)

yksom/Cargo.toml:
	git clone --recursive $(YKSOM_REPO) yksom
	cd yksom && git checkout $(YKSOM_VERSION)

clean-benchmarks:
	rm -rf results

clean-builds:
	rm -rf yksom
	rm -rf som-rs
	rm -rf $(BIN)

clean: clean-benchmarks clean-builds
