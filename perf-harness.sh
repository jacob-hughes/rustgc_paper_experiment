#!/bin/sh

CFG=$1
INVOCATION=$2
BENCHMARK_SUITE=$3
BIN=$4
BENCHMARK=$5
shift 5

OUTDIR="$(pwd)/results/$EXPERIMENT/$BENCHMARK_SUITE/perf/metrics"
LOGFILE="$BIN.$INVOCATION.$EXPERIMENT-$CFG.$BENCHMARK.csv"
export ALLOY_LOG="$OUTDIR/runtime/$LOGFILE"

if [ "$EXPERIMENT" == "gcvs" ] && [ "$CFG" != "gc" ]; then
    export GC_DONT_GC=true
fi

"benchmarks/$BENCHMARK_SUITE/bin/$EXPERIMENT/$CFG/perf/bin/$BIN" "$@"
