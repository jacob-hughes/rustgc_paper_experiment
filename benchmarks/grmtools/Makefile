PWD != pwd

GRMTOOLS_SRC = $(PWD)/grmtools
GRMTOOLS_REPO = https://github.com/softdevteam/grmtools
GRMTOOLS_VERSION = a0972be0777e599a3dbca710fb0a595c39560b69

CACTUS_SRC = $(PWD)/cactus
CACTUS_REPO = https://github.com/softdevteam/cactus
CACTUS_VERSION = 8d34c207e1479cecf0b9b2f7beb1a0c22c8949ad

REGEX_SRC = $(PWD)/regex
REGEX_REPO = https://github.com/rust-lang/regex
REGEX_VERSION = bcbe40342628b15ab2543d386c745f7f0811b791

HADOOP_SRC = $(PWD)/hadoop
ECLIPSE_SRC = $(PWD)/eclipse
SPRING_SRC = $(PWD)/spring
JENKINS_SRC = $(PWD)/jenkins

HADOOP_REPO = https://github.com/apache/hadoop
ECLIPSE_REPO = https://github.com/eclipse-platform/eclipse.platform
SPRING_REPO = https://github.com/spring-projects/spring-framework
JENKINS_REPO = https://github.com/jenkinsci/jenkins

JAVA_SRCS = $(HADOOP_SRC) $(ECLIPSE_SRC) $(SPRING_SRC) $(JENKINS_SRC)
BUILD_SRCS = $(GRMTOOLS_SRC) $(CACTUS_SRC) $(REGEX_SRC)

GCVS_CFGS = gcvs/gc gcvs/rc
GCVS_CFGS := $(addsuffix /perf, $(GCVS_CFGS)) $(addsuffix /mem, $(GCVS_CFGS))

PATCH = $(PWD)/patches

BINS = java_parse
RESULTS = $(foreach e,$(EXPERIMENTS),$(foreach bin,$(BINS),$(e)/$(bin)))
RESULTS := $(addprefix $(RESULTS_DIR)/, $(addsuffix /perf.csv, $(RESULTS)) $(addsuffix /mem.csv, $(RESULTS)))
PLOTS = $(dir $(patsubst $(RESULTS_DIR)/%.csv, $(PLOTS_DIR)/%, $(RESULTS)))

.PHONY: all build bench clean

all: venv build bench plot

bench: bench-perf bench-barriers bench-elision

define patch_repo
	cd $1 && git reset --hard
	$(eval P := $(wildcard $(PATCH)/$(notdir $1).$(subst /,.,$(dir $2))diff))
	$(if $(P),\
		cd $1 && git apply $(P),)
endef

build: srcs $(GCVS_CFGS)
	@echo "hello"

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

$(GCVS_CFGS):
	$(call patch_repo,$(CACTUS_SRC),$@)

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


srcs: $(JAVA_SRCS) $(BUILD_SRCS)

$(HADOOP_SRC):
	git clone $(HADOOP_REPO) $@ --depth 1

$(ECLIPSE_SRC):
	git clone $(ECLIPSE_REPO) $@ --depth 1

$(SPRING_SRC):
	git clone $(SPRING_REPO) $@ --depth 1

$(JENKINS_SRC):
	git clone $(JENKINS_REPO) $@ --depth 1

$(GRMTOOLS_SRC):
	git clone $(GRMTOOLS_REPO) $@
	cd $@ && git checkout $(GRMTOOLS_VERSION)
	cd $@ && git apply $(PATCH)/grmtools.diff

$(CACTUS_SRC):
	git clone $(CACTUS_REPO) $@
	cd $@ && git checkout $(CACTUS_VERSION)

$(REGEX_SRC):
	git clone $(CACTUS_REPO) $@
	cd $@ && git checkout $(CACTUS_VERSION)

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

clean:
	rm -rf $(BUILD_SRCS)
