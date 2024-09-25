#! /bin/sh

set -e
ALLOYSV=f07e460fe8bfba7490bdf62870326d634862d1bf
ALLOY_DIR="alloy"
ALLOY_PATCH_DIR=../configs/alloy

# Alloy configurations
ALLOY_BARRIERS_OPT=barriers_opt
ALLOY_BARRIERS_NONE=barriers_none
ALLOY_BARRIERS_NAIVE=barriers_naive

ALLOY_FINALISATION_NAIVE=finalisation_naive
ALLOY_FINALISATION_OPT=finalisation_opt

ALLOY_RELEASE=release

BOOTSTRAP_STAGE=1

SOMRSSV=35b780cbee765cca24201fe063d3f1055ec7f608
SOMRS_DIR="som-rs"
SOMRS_PATCH_DIR=../configs/som-rs

YKSOMSV=fc7c7c131ba93b7e3c85a172fbcc245f29c324d6
WLAMBDASV=b370e342fdffeae6f

EXP_RESULTS_DIR=raw_data

mk_alloy_cfg() {
    prefix=$1
    shift
    if [ -f "${ALLOY_DIR}/${prefix}.lock" ]; then
	echo "WARNING: '${ALLOY_DIR}/${prefix}' already exists. Skipping build."
	echo "WARNING: To force a rebuild, remove the '${ALLOY_DIR}/${prefix}.lock' file."
	return
    fi
    (cd ${ALLOY_DIR} && \
	git reset --hard && \
	for patch in "$@"
	do
	    git apply "${ALLOY_PATCH_DIR}/${patch}"
	    echo "Applied patch: ${patch}"
	done && \
	python3 x.py install --config ../benchmark.config.toml \
	    --stage ${BOOTSTRAP_STAGE} \
	    --set build.docs=false \
	    --set install.prefix=${prefix} \
	    --set install.sysconfdir=etc && \
	touch ${prefix}.lock && \
	git reset --hard
    )
}

mk_somrs_cfg() {
    cfg=$1
    build_dir=$2
    shift
    shift
    rustc="../${ALLOY_DIR}/${cfg}/bin/rustc"
    if [ -f "${SOMRS_DIR}/${build_dir}.lock" ]; then
	echo "WARNING: '${SOMRS_DIR}/${build_dir}' already exists. Skipping build."
	echo "WARNING: To force a rebuild, remove the 'som-rs/${build_dir}.lock' file."
	return
    fi
    (cd ${SOMRS_DIR} && \
	git reset --hard && \
	for patch in "$@"
	do
	    echo "Applying patch: ${patch}"
	    git apply "${SOMRS_PATCH_DIR}/${patch}"
	done && \
	RUSTC="$rustc" cargo build --release -p som-interpreter-bc --target-dir=${build_dir} && \
	touch ${build_dir}.lock && \
	echo "INFO: som-rs compiled with '${rustc}'" && \
	git reset --hard
    )
}

mk_early_finaliser_exp() {
    mk_alloy_cfg $ALLOY_BARRIERS_OPT "disable_finaliser_elision.patch"
    mk_alloy_cfg $ALLOY_BARRIERS_NAIVE "disable_finaliser_elision.patch" "all_barriers.patch"
    mk_alloy_cfg $ALLOY_BARRIERS_NONE "disable_finaliser_elision.patch" "no_barriers.patch"

    mk_somrs_cfg $ALLOY_BARRIERS_OPT $ALLOY_BARRIERS_OPT "use_gc.patch"
    mk_somrs_cfg $ALLOY_BARRIERS_NAIVE $ALLOY_BARRIERS_NAIVE "use_gc.patch"
    mk_somrs_cfg $ALLOY_BARRIERS_NONE $ALLOY_BARRIERS_NONE "use_gc.patch"
}

mk_finaliser_elision_exp() {
    mk_alloy_cfg $ALLOY_FINALISATION_NAIVE "disable_finaliser_elision.patch"
    mk_alloy_cfg $ALLOY_FINALISATION_OPT

    mk_somrs_cfg \
	$ALLOY_FINALISATION_NAIVE \
	$ALLOY_FINALISATION_NAIVE \
	"use_gc.patch" \
	"naive_finalisation.patch"

    mk_somrs_cfg \
	$ALLOY_FINALISATION_OPT \
	$ALLOY_FINALISATION_OPT \
	"use_gc.patch" \
	"optimised_finalisation.patch"
}

mk_som_rs_perf_exp() {
    mk_alloy_cfg $ALLOY_RELEASE

    mk_somrs_cfg $ALLOY_RELEASE "rc" "bdwgc_allocator.patch"
    mk_somrs_cfg $ALLOY_RELEASE "gc" "use_gc.patch"
}

run_somrs_exp() {
    expname=$1
    shift
    (cd ${SOMRS_DIR} && \
	git reset --hard && \
	git apply "${SOMRS_PATCH_DIR}/tune_benchmarks.patch" && \
	for patch in "$@"
	do
	    echo "Applying patch: ${patch}"
	    git apply "${SOMRS_PATCH_DIR}/${patch}"
	done && \
	rebench -R -D --experiment all rebench.conf all
    )
    mv "${SOMRS_DIR}/rebench.data" ${expname}
}

run_early_finaliser_exp() {
    mk_early_finaliser_exp
    resultsfile="${EXP_RESULTS_DIR}/som_rs_barriers.data"
    run_somrs_exp ${resultsfile} "rebench_barriers.patch"
    python3 process_graph.py ${resultsfile} "som_rs_barriers.svg"
}

run_finaliser_elision_exp() {
    mk_finaliser_elision_exp
    mkdir -p som-rs/naive_counts
    mkdir -p som-rs/elision_counts
    resultsfile="${EXP_RESULTS_DIR}/som_rs_finaliser_elision.data"
    run_somrs_exp ${resultsfile} "rebench_finaliser_elision.patch"
    python3 process_graph.py ${resultsfile} "som_rs_finaliser_elision.svg"
}

run_perf_exp() {
    mk_som_rs_perf_exp
    resultsfile="${EXP_RESULTS_DIR}/som_rs_perf.data"
    run_somrs_exp ${resultsfile} "rebench_perf.patch"
    python3 process_graph.py ${resultsfile} "som_rs_perf.svg"
}

# ============= Build Alloy with each configuration =====================
if [ ! -d ${ALLOY_DIR} ]; then
    git clone https://github.com/softdevteam/alloy ${ALLOY_DIR}
    (cd ${ALLOY_DIR} && git checkout ${ALLOYSV})
fi

if [ ! -d ${SOMRS_DIR} ]; then
    git clone --recursive https://github.com/Hirevo/som-rs ${SOMRS_DIR}
    (cd ${SOMRS_DIR} && git checkout ${SOMRSSV})
fi

# run_early_finaliser_exp
run_finaliser_elision_exp
run_perf_exp

