#!/bin/env bash
export PATH="$(dirname $0)/:$PATH"
CMD_DIR="benchmarks/ripgrep/commands"
LINUX="benchmarks/ripgrep/linux"

CMD=$(<$CMD_DIR/$@)

if [ -z "${HT}" ]; then
    $CMD $LINUX
else
    $HT $CMD $LINUX
fi
