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
PB_SRC = $(PWD)/parserbench

GCVS_CFGS = gcvs/gc gcvs/rc
GCVS_CFGS := $(addsuffix /perf, $(GCVS_CFGS)) $(addsuffix /mem, $(GCVS_CFGS))

PATCH = $(PWD)/patches

RESULTS = $(foreach e,$(EXPERIMENTS),$(foreach bin,grmtools,$(e)/$(bin)))
RESULTS := $(addprefix $(RESULTS_DIR)/, $(addsuffix /perf.csv, $(RESULTS)) $(addsuffix /mem.csv, $(RESULTS)))
PLOTS = $(dir $(patsubst $(RESULTS_DIR)/%.csv, $(PLOTS_DIR)/%, $(RESULTS)))

.PHONY: all build bench clean

all: venv build bench plot

define patch_repo
	cd $1 && git reset --hard
	$(eval P := $(wildcard $(PATCH)/$(notdir $1).$(subst /,.,$(dir $2))diff))
	$(if $(P),\
		cd $1 && git apply $(P),)
endef

define build_parserbench
	@echo "Building parserbench with RUSTC=$2"
	RUSTFLAGS="-L $(LIBGC_PATH)" RUSTC="$2" cargo install \
		--path $(PB_SRC) \
		--target-dir $(PB_SRC)/build/$@ \
		--root $(PB_SRC)/$@
endef

build: srcs $(GCVS_CFGS) $(ALLOY_CFGS)
	@echo "hello"

$(GCVS_CFGS):
	$(call patch_repo,$(CACTUS_SRC),$@)
	$(call patch_repo,$(REGEX_SRC),$@)
	$(call build_parserbench,$@,$(ALLOY_PATH)/gcvs/$(notdir $@)/bin/rustc)

$(ALLOY_CFGS):
	cd $(CACTUS_SRC) && git reset --hard && \
		git apply $(PATCH)/cactus.alloy.diff
	cd $(REGEX_SRC) && git reset --hard && \
		git apply $(PATCH)/regex.alloy.diff
	$(call build_parserbench,$@,$(ALLOY_PATH)/$@/bin/rustc)

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
	git clone $(REGEX_REPO) $@
	cd $@ && git checkout $(REGEX_VERSION)

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
	rm -rf $(BUILD_SRCS)
