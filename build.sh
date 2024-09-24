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

BOOTSTRAP_STAGE=1

SOMRSSV=35b780cbee765cca24201fe063d3f1055ec7f608
SOMRS_DIR="som-rs"
SOMRS_PATCH_DIR=../configs/som-rs

YKSOMSV=fc7c7c131ba93b7e3c85a172fbcc245f29c324d6
WLAMBDASV=b370e342fdffeae6f

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
	done && \
	python3 x.py install --config ../benchmark.config.toml \
	    --stage ${BOOTSTRAP_STAGE} \
	    --set build.docs=false \
	    --set install.prefix=${prefix} \
	    --set install.sysconfdir=etc && \
	touch ${prefix}.lock
    )
}

mk_somrs_cfg() {
    prefix=$1
    shift
    rustc="../${ALLOY_DIR}/${prefix}/bin/rustc"
    if [ -f "${SOMRS_DIR}/${prefix}.lock" ]; then
	echo "WARNING: '${SOMRS_DIR}/${prefix}' already exists. Skipping build."
	echo "WARNING: To force a rebuild, remove the 'som-rs/${prefix}.lock' file."
	return
    fi
    (cd ${SOMRS_DIR} && \
	git reset --hard && \
	for patch in "$@"
	do
	    git apply "${SOMRS_PATCH_DIR}/${patch}"
	done && \
	RUSTC="$rustc" cargo build --release -p som-interpreter-bc --target-dir=${prefix} &&
	touch ${prefix}.lock
    )
}

mk_early_finaliser_exp() {
    mk_alloy_cfg $ALLOY_BARRIERS_OPT "disable_finaliser_elision.patch"
    mk_somrs_cfg $ALLOY_BARRIERS_OPT "use_gc.patch"

    mk_alloy_cfg $ALLOY_BARRIERS_NAIVE "disable_finaliser_elision.patch" "all_barriers.patch"
    mk_somrs_cfg $ALLOY_BARRIERS_NAIVE "use_gc.patch"

    mk_alloy_cfg $ALLOY_BARRIERS_NONE "disable_finaliser_elision.patch" "no_barriers.patch"
    mk_somrs_cfg $ALLOY_BARRIERS_NONE "use_gc.patch"
}

# ============= Build Alloy with each configuration =====================
mk_early_finaliser_exp

