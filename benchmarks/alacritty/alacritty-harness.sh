#!/bin/env sh
BIN="$(dirname $0)/alacritty"
BM_DIR="benchmarks/alacritty/vtebench/benchmarks/$@"

if [ -e "$BM_DIR/setup" ]; then
    CMD="($BM_DIR/setup;$BM_DIR/benchmark)"
else
    CMD=$BM_DIR/benchmark
fi

export DISPLAY=":99"

if [ -z "${HT}" ]; then
    $BIN -e /bin/bash -c $CMD
else
    $HT $BIN -e /bin/bash -c $CMD
fi
