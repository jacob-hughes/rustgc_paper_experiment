PWD != pwd

SOMRS_REPO = https://github.com/Hirevo/som-rs
SOMRS_VERSION = 35b780cbee765cca24201fe063d3f1055ec7f608
SOMRS_SRC = $(PWD)/som-rs

YKSOM_REPO = https://github.com/softdevteam/yksom
YKSOM_VERSION = master
YKSOM_SRC = $(PWD)/yksom

GCVS_CFGS = gcvs/gc gcvs/rc
GCVS_CFGS := $(addsuffix /perf, $(GCVS_CFGS)) $(addsuffix /mem, $(GCVS_CFGS))

PATCH = $(PWD)/patches

BINS = som-rs-ast som-rs-bc yksom
RESULTS = $(foreach e,$(EXPERIMENTS),$(foreach bin,$(BINS),$(e)/$(bin)))
RESULTS := $(addprefix $(RESULTS_DIR)/, $(addsuffix /perf.csv, $(RESULTS)) $(addsuffix /mem.csv, $(RESULTS)))
PLOTS = $(dir $(patsubst $(RESULTS_DIR)/%.csv, $(PLOTS_DIR)/%, $(RESULTS)))

all: build

.PHONY: all build clean

build: $(SOMRS_SRC) $(YKSOM_SRC) $(ALLOY_CFGS) $(GCVS_CFGS)

define build_somrs
	@echo "Building som-rs with RUSTC=$2"
	RUSTFLAGS="-L $(LIBGC_PATH)" RUSTC="$2" cargo install --path $(SOMRS_SRC)/som-interpreter-ast \
		--target-dir $(SOMRS_SRC)/som-interpreter-ast/build/$1 \
		--root $(SOMRS_SRC)/$1
	ln -s $(SOMRS_SRC)/$1/bin/som-interpreter-ast \
		$(SOMRS_SRC)/$1/bin/som-rs-ast
	RUSTFLAGS="-L $(LIBGC_PATH)" RUSTC="$2" cargo install --path $(SOMRS_SRC)/som-interpreter-bc \
		--target-dir $(SOMRS_SRC)/som-interpreter-bc/build/$1 \
		--root $(SOMRS_SRC)/$1
	ln -s $(SOMRS_SRC)/$1/bin/som-interpreter-bc \
		$(SOMRS_SRC)/$1/bin/som-rs-bc
endef

define build_yksom
	@echo "Building yksom with RUSTC=$2"
	RUSTFLAGS="-L $(LIBGC_PATH)" RUSTC="$2" cargo install --path $(YKSOM_SRC) \
		--target-dir $(YKSOM_SRC)/build/$1 \
		--root $(YKSOM_SRC)/$1
endef

$(GCVS_CFGS):
	cd $(SOMRS_SRC) && git reset --hard && \
	git apply $(PATCH)/som-rs.$(subst /,.,$(dir $@))diff
	$(call build_somrs,$@,$(ALLOY_PATH)/gcvs/$(notdir $@)/bin/rustc)
	$(call build_yksom,$@,$(ALLOY_PATH)/gcvs/$(notdir $@)/bin/rustc)

$(ALLOY_CFGS):
	cd $(SOMRS_SRC) && git reset --hard && \
	git apply $(PATCH)/som-rs.alloy.diff
	$(call build_somrs,$@,$(ALLOY_PATH)/$@/bin/rustc)
	$(call build_yksom,$@,$(ALLOY_PATH)/$@/bin/rustc)

$(SOMRS_SRC):
	git clone --recursive $(SOMRS_REPO) $@
	cd $@ && git checkout $(SOMRS_VERSION)
	ln -s $(SOMRS_SRC)/core-lib $(SOMRS_SRC)/SOM

$(YKSOM_SRC):
	git clone --recursive $(YKSOM_REPO) $@
	cd $@ && git checkout $(YKSOM_VERSION)


bench: $(RESULTS)

$(RESULTS_DIR)/%.csv:
	mkdir -p $(dir $@){metrics,heaptrack,samples}
	- $(REBENCH_EXEC) -R -D \
		--invocations $(PEXECS) \
		--iterations 1 \
		-df $@ $(PWD)/rebench.conf $(subst /,-,$*)

plot: $(PLOTS)

$(PLOTS_DIR)/%:
	mkdir -p $@/mem
	$(REBENCH_PROCESSOR) $@ ../../results.bk/$(dir $*)

clean-plots:
	$(foreach plot, $(PLOTS_DIR), rm -rf $(plot);)

clean-benchmarks:
	$(foreach r, $(RESULTS), rm -rf $(r) $(r)/metrics $(r)/heaptrack $(r)/samples;)

clean:
	rm -rf $(SOMRS_SRC) $(YKSOM_SRC)
