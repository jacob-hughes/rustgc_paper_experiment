#!/bin/sh

CFG=$1
INVOCATION=$2
shift 2

BENCHMARK=$@
OUTDIR="../../results/$EXPERIMENT/grmtools"

BIN="$(pwd)/parserbench/$EXPERIMENT/$CFG/$EXPTYPE/bin/parserbench $@/"

export LD_LIBRARY_PATH="../../bdwgc/lib"

if [ "$CFG" = "rc" ]; then
    export GC_DONT_GC=true
fi

if [ "$EXPTYPE" = "perf" ]; then
    METRICS_LOGFILE="$OUTDIR/metrics/$INVOCATION.$EXPERIMENT-$CFG.$BENCHMARK.csv"
    ALLOY_LOG="$METRICS_LOGFILE" $BIN
else
    HTPATH="../../heaptrack/bin"
    SAMPLER="../../venv/bin/python ../../sample_memory.py"
    HTDATA="$OUTDIR/heaptrack/$INVOCATION.$EXPERIMENT-$CFG.$BENCHMARK"
    SAMPLERDATA="$OUTDIR/samples/$INVOCATION.$EXPERIMENT-$CFG.$BENCHMARK.csv"
    $HTPATH/heaptrack --record-only -o $HTDATA $BIN
    $HTPATH/heaptrack_print -M $HTDATA.massif $HTDATA.zst
    $SAMPLER -o $SAMPLERDATA $BIN
fi;

