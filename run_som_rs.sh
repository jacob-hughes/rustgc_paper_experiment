#! /bin/sh

set -e

export PEXECS=${PEXECS:-10}
export ITERS=${ITERS:-10}

EXP_RESULTS_DIR=results
SOM_RS_RCONF=som_rs.conf

run_exp() {
    exp=${1}
    expdir="${EXP_RESULTS_DIR}/som_rs/${1}"
    rdata="${expdir}/${1}.data"
    [ -e ${expdir} ] && rm -rf ${expdir}
    mkdir -p ${expdir}
    rebench -R -D \
	--invocations $PEXECS \
	--iterations $ITERS \
	-df "${rdata}" \
	${SOM_RS_RCONF} ${exp} || true
}

usage() {
    echo "Run som-rs experiments."
    echo "usage: $0 perf | barriers | elision"
}

if [ $# -ne 1 ]; then
    usage
    exit 1
fi

case "$1" in
  "perf"|"barriers"|"elision")
    run_exp $1
    ;;
  *)
    usage
    exit 1
    ;;
esac
