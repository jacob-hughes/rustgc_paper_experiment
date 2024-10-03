#! /bin/sh

set -e

SOMRSSV=35b780cbee765cca24201fe063d3f1055ec7f608
SOMRS_DIR="som-rs"
SOMRS_PATCH_DIR=../configs/som-rs

# Alloy configurations
ALLOY_BARRIERS_OPT=barriers_opt
ALLOY_BARRIERS_NONE=barriers_none
ALLOY_BARRIERS_NAIVE=barriers_naive

ALLOY_FINALISATION_NAIVE=finalisation_naive
ALLOY_FINALISATION_OPT=finalisation_opt

ALLOY_RELEASE=release
ALLOY_DIR=alloy

if [ ! -d ${SOMRS_DIR} ]; then
    git clone --recursive https://github.com/Hirevo/som-rs ${SOMRS_DIR}
    (cd ${SOMRS_DIR} && git checkout ${SOMRSSV})
fi

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

usage() {
    echo "Build som-rs configurations for experiments."
    echo "usage: $0 perf | barriers | elision"
}

if [ $# -ne 1 ]; then
    usage
elif [ "$@" = "perf" ]; then
    mk_somrs_cfg $ALLOY_RELEASE "perf_rc" "bdwgc_allocator.patch"
    mk_somrs_cfg $ALLOY_RELEASE "perf_gc" "use_gc.patch"
elif [ "$@" = "barriers" ]; then
    mk_somrs_cfg $ALLOY_BARRIERS_OPT $ALLOY_BARRIERS_OPT "count_finalisers.patch" "use_gc.patch"
    mk_somrs_cfg $ALLOY_BARRIERS_NAIVE $ALLOY_BARRIERS_NAIVE "count_finalisers.patch" "use_gc.patch"
    mk_somrs_cfg $ALLOY_BARRIERS_NONE $ALLOY_BARRIERS_NONE "count_finalisers.patch" "use_gc.patch"
elif [ "$@" = "elision" ]; then
    mk_somrs_cfg $ALLOY_FINALISATION_NAIVE $ALLOY_FINALISATION_NAIVE \
	"count_finalisers.patch" "use_gc.patch"

    mk_somrs_cfg $ALLOY_FINALISATION_OPT $ALLOY_FINALISATION_OPT \
	"count_finalisers.patch" "use_gc.patch"
else
    usage
fi
