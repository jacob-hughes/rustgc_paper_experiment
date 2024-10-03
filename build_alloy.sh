#! /bin/sh

set -e

ALLOY_DIR="alloy"
BDWGC_DIR="bdwgc"
ALLOY_PATCH_DIR=../configs/alloy

ALLOYSV=c304b5dfe0631d386739d99adb0df4255e1793e1
BDWGCSV=4f65865f7e84f66ae730658123cd63a9490bd766

# Alloy configurations
ALLOY_BARRIERS_OPT=barriers_opt
ALLOY_BARRIERS_NONE=barriers_none
ALLOY_BARRIERS_NAIVE=barriers_naive
ALLOY_FINALISATION_NAIVE=finalisation_naive
ALLOY_FINALISATION_OPT=finalisation_opt
ALLOY_DYN=release_dyn
ALLOY_RELEASE=release

BOOTSTRAP_STAGE=1

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

mk_bdwgc() {
    (cd ${BDWGC_DIR} && \
	mkdir -p out && \
	cd out && \
	cmake -DCMAKE_BUILD_TYPE=Release \
	      -Denable_valgrind_tracking=On \
	      -Denable_parallel_mark=Off \
	      ../ && \
	make -j$(nproc)
    )
}

if [ ! -d ${ALLOY_DIR} ]; then
    git clone https://github.com/softdevteam/alloy ${ALLOY_DIR}
    (cd ${ALLOY_DIR} && git checkout ${ALLOYSV})
fi

if [ ! -d ${BDWGC_DIR} ]; then
    git clone --recursive https://github.com/softdevteam/bdwgc ${BDWGC_DIR}
    (cd ${BDWGC_DIR} && git checkout ${BDWGCSV})
fi

# ============= Build ALLOY with each configuration =====================
# mk_alloy_cfg $ALLOY_RELEASE

# mk_alloy_cfg $ALLOY_BARRIERS_OPT "disable_finaliser_elision.patch"
# mk_alloy_cfg $ALLOY_BARRIERS_NAIVE "disable_finaliser_elision.patch" "all_barriers.patch"
# mk_alloy_cfg $ALLOY_BARRIERS_NONE "disable_finaliser_elision.patch" "no_barriers.patch"

# mk_alloy_cfg $ALLOY_BARRIERS_OPT "disable_finalisers.patch"
# mk_alloy_cfg $ALLOY_BARRIERS_NAIVE "disable_finaliser.patch" "all_barriers.patch"
# mk_alloy_cfg $ALLOY_BARRIERS_NONE "disable_finaliser.patch" "no_barriers.patch"

# mk_alloy_cfg $ALLOY_FINALISATION_NAIVE "disable_finaliser_elision.patch"
# mk_alloy_cfg $ALLOY_FINALISATION_OPT

# mk_bdwgc
# LD_LIBRARY_PATH="/home/jake/research/rustgc_paper_experiment/bdwgc/out" \
#     RUSTFLAGS="-L /home/jake/research/rustgc_paper_experiment/bdwgc/out" \
#     GC_LINK_DYNAMIC=true mk_alloy_cfg $ALLOY_BARRIERS_OPT "disable_finaliser_elision.patch"
#
# LD_LIBRARY_PATH="/home/jake/research/rustgc_paper_experiment/bdwgc/out" \
#     RUSTFLAGS="-L /home/jake/research/rustgc_paper_experiment/bdwgc/out" \
#     GC_LINK_DYNAMIC=true mk_alloy_cfg $ALLOY_BARRIERS_NAIVE "disable_finaliser_elision.patch" "all_barriers.patch"
# #
# LD_LIBRARY_PATH="/home/jake/research/rustgc_paper_experiment/bdwgc/out" \
#     RUSTFLAGS="-L /home/jake/research/rustgc_paper_experiment/bdwgc/out" \
#     GC_LINK_DYNAMIC=true mk_alloy_cfg $ALLOY_BARRIERS_NONE "disable_finaliser_elision.patch" "no_barriers.patch"
# LD_LIBRARY_PATH="/home/jake/research/rustgc_paper_experiment/bdwgc/out" \
#     RUSTFLAGS="-L /home/jake/research/rustgc_paper_experiment/bdwgc/out" \
#     mk_somrs_cfg "alloy_dyn" "alloy_dyn"

usage() {
    echo "Build Alloy configurations for experiments."
    echo "usage: build.sh perf | barriers | elision"
}

if [ $# -ne 1 ]; then
    usage
elif [ "$@" = "perf" ]; then
    echo "making perf"
    mk_alloy_cfg $ALLOY_RELEASE
elif [ "$@" = "barriers" ]; then
    mk_alloy_cfg $ALLOY_BARRIERS_OPT "disable_finaliser_elision.patch"
    mk_alloy_cfg $ALLOY_BARRIERS_NAIVE "disable_finaliser_elision.patch" "all_barriers.patch"
    mk_alloy_cfg $ALLOY_BARRIERS_NONE "disable_finaliser_elision.patch" "no_barriers.patch"
elif [ "$@" = "elision" ]; then
    mk_alloy_cfg $ALLOY_FINALISATION_NAIVE "disable_finaliser_elision.patch"
    mk_alloy_cfg $ALLOY_FINALISATION_OPT
elif [ "$@" = "mem" ]; then
    echo "making mem"
    mk_bdwgc
    LD_LIBRARY_PATH="bdwgc/out" RUSTFLAGS="-L bdwgc/out" GC_LINK_DYNAMIC=true mk_alloy_cfg \
	$ALLOY_BARRIERS_OPT "disable_finaliser_elision.patch"
else
    usage
fi
