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

BDWGC_DIR="bdwgc"
BDWGCSV=a69f8b7601c2309107b58c1f70f73e4b9f8f3421

SOMRSSV=35b780cbee765cca24201fe063d3f1055ec7f608
SOMRS_DIR="som-rs"
SOMRS_PATCH_DIR=../configs/som-rs

YKSOMSV=fc7c7c131ba93b7e3c85a172fbcc245f29c324d6
YKSOM_DIR="yksom"
YKSOM_PATCH_DIR=../configs/yksom


WLAMBDASV=b370e342fdffeae6f

EXP_RESULTS_DIR=raw_data

PEXECS=3
ITERS=5

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

mk_yksom_cfg() {
    cfg=$1
    build_dir=$2
    shift
    shift
    rustc="../${ALLOY_DIR}/${cfg}/bin/rustc"
    if [ -f "${YKSOM_DIR}/${build_dir}.lock" ]; then
	echo "WARNING: '${YKSOM_DIR}/${build_dir}' already exists. Skipping build."
	echo "WARNING: To force a rebuild, remove the 'yksom/${build_dir}.lock' file."
	return
    fi
    (cd ${YKSOM_DIR} && \
	git reset --hard && \
	for patch in "$@"
	do
	    echo "Applying patch: ${patch}"
	    git apply "${YKSOM_PATCH_DIR}/${patch}"
	done && \
	RUSTC="$rustc" cargo build --release --target-dir=${build_dir} && \
	touch ${build_dir}.lock && \
	echo "INFO: yksom compiled with '${rustc}'" && \
	git reset --hard
    )
}

run_exp() {
    vm=$1
    exp=${2}
    expname="${vm}_${exp}"
    expdir="${EXP_RESULTS_DIR}/${expname}"
    rebench_datafile="${expdir}/${expname}.data"
    echo "$expdir"
    [ -e ${expdir} ] && rm -rf ${expdir}
    mkdir -p ${expdir}
    rebench -R -D \
	--invocations $PEXECS \
	--iterations $ITERS \
	-df "${rebench_datafile}" \
	"${vm}.conf" ${exp} || true
    (cd ${expdir} && \
	python3 ../../process_graph.py "${expname}.data" "${expname}.svg")
}

if [ ! -d ${ALLOY_DIR} ]; then
    git clone https://github.com/softdevteam/alloy ${ALLOY_DIR}
    (cd ${ALLOY_DIR} && git checkout ${ALLOYSV})
fi

if [ ! -d ${SOMRS_DIR} ]; then
    git clone --recursive https://github.com/Hirevo/som-rs ${SOMRS_DIR}
    (cd ${SOMRS_DIR} && git checkout ${SOMRSSV})
fi

if [ ! -d ${BDWGC_DIR} ]; then
    git clone --recursive https://github.com/softdevteam/bdwgc ${BDWGC_DIR}
    (cd ${BDWGC_DIR} && git checkout ${BDWGCSV})
fi

if [ ! -d ${YKSOM_DIR} ]; then
    git clone --recursive https://github.com/softdevteam/yksom ${YKSOM_DIR}
    (cd ${YKSOM_DIR} && git checkout ${YKSOMSV})
fi

# ============= Build ALLOY with each configuration =====================
mk_alloy_cfg $ALLOY_RELEASE

mk_alloy_cfg $ALLOY_BARRIERS_OPT "disable_finaliser_elision.patch"
mk_alloy_cfg $ALLOY_BARRIERS_NAIVE "disable_finaliser_elision.patch" "all_barriers.patch"
mk_alloy_cfg $ALLOY_BARRIERS_NONE "disable_finaliser_elision.patch" "no_barriers.patch"

mk_alloy_cfg $ALLOY_FINALISATION_NAIVE "disable_finaliser_elision.patch"
mk_alloy_cfg $ALLOY_FINALISATION_OPT

mk_bdwgc
LD_LIBRARY_PATH="/home/jake/research/rustgc_paper_experiment/bdwgc/out" \
    RUSTFLAGS="-L /home/jake/research/rustgc_paper_experiment/bdwgc/out" \
    GC_LINK_DYNAMIC=true mk_alloy_cfg "alloy_dyn"

LD_LIBRARY_PATH="/home/jake/research/rustgc_paper_experiment/bdwgc/out" \
    RUSTFLAGS="-L /home/jake/research/rustgc_paper_experiment/bdwgc/out" \
    mk_somrs_cfg "alloy_dyn" "alloy_dyn"

# ============= Build SOM_RS with each configuration =====================
mk_somrs_cfg $ALLOY_RELEASE "perf_rc" "bdwgc_allocator.patch"
mk_somrs_cfg $ALLOY_RELEASE "perf_gc" "use_gc.patch"

mk_somrs_cfg $ALLOY_FINALISATION_NAIVE $ALLOY_FINALISATION_NAIVE \
    "count_finalisers.patch" "use_gc.patch"

mk_somrs_cfg $ALLOY_FINALISATION_OPT $ALLOY_FINALISATION_OPT \
    "count_finalisers.patch" "use_gc.patch"

mk_somrs_cfg $ALLOY_BARRIERS_OPT $ALLOY_BARRIERS_OPT "count_finalisers.patch" "use_gc.patch"
mk_somrs_cfg $ALLOY_BARRIERS_NAIVE $ALLOY_BARRIERS_NAIVE "count_finalisers.patch" "use_gc.patch"
mk_somrs_cfg $ALLOY_BARRIERS_NONE $ALLOY_BARRIERS_NONE "count_finalisers.patch" "use_gc.patch"

# ============= Build YKSOM with each configuration =====================
mk_yksom_cfg $ALLOY_FINALISATION_OPT $ALLOY_FINALISATION_OPT
mk_yksom_cfg $ALLOY_FINALISATION_NAIVE $ALLOY_FINALISATION_NAIVE

mk_yksom_cfg $ALLOY_BARRIERS_NAIVE $ALLOY_BARRIERS_NAIVE
mk_yksom_cfg $ALLOY_BARRIERS_NONE $ALLOY_BARRIERS_NONE
mk_yksom_cfg $ALLOY_BARRIERS_OPT $ALLOY_BARRIERS_OPT

# ============= Run experiments =========================================

run_exp "som_rs" "barriers"
run_exp "som_rs" "finaliser_elision"
run_exp "som_rs" "perf"

run_exp "yksom" "barriers"
run_exp "yksom" "finaliser_elision"
