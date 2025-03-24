#!/bin/env bash

CFG=$1
INVOCATION=$2
BENCHMARK_SUITE=$3
BIN=$4
BENCHMARK=$5
shift 5

OUTDIR="$(pwd)/results/$EXPERIMENT/$BENCHMARK_SUITE/metrics"

if [ "$EXPERIMENT" == "gcvs" ] && [ "$CFG" != "gc" ]; then
    export GC_DONT_GC=true
fi

SAMPLER="venv/bin/python sample_memory.py"
SAMPLERDATA="$OUTDIR/rss/$BIN.$INVOCATION.$EXPERIMENT-$CFG.$BENCHMARK.csv"

HTPATH="heaptrack/bin"
HTDATA="$OUTDIR/heaptrack/$BIN.$INVOCATION.$EXPERIMENT-$CFG.$BENCHMARK"

LOGFILE="$BIN.$INVOCATION.$EXPERIMENT-$CFG.$BENCHMARK.csv"

export LD_LIBRARY_PATH="bdwgc/lib"
export HT="$HTPATH/heaptrack --record-only -o $HTDATA"
export ALLOY_LOG="$OUTDIR/runtime/$LOGFILE"

(./benchmarks/$BENCHMARK_SUITE/bin/$EXPERIMENT/$CFG/mem/bin/$BIN "$@" 2>&1) |
    grep -A3 'heaptrack stats' | tail -n 3 | tr -d "[:blank:]" |
    awk -F: '
    BEGIN {print "allocations,leaked allocations,temporary allocations"}
    {values = values","$2}
    END {print substr(values,2)}' >$HTDATA.summary.csv

$HTPATH/heaptrack_print -M $HTDATA.massif $HTDATA.zst
