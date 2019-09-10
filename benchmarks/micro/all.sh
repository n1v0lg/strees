#!/usr/bin/env bash

# Ops to benchmark
declare -a OPS=(
    "shuffle"
)

# Default benchmark sizes
declare -a SIZES=(
    16
    128
    1024
)

for OP in "${OPS[@]}";
do
    for SIZE in "${SIZES[@]}";
    do
        # TODO hack hack hack
        bash run.sh ${SIZE} ${OP} --debug > /dev/null 2>&1;
        if [ "$?" -ne 0 ]; then
            echo "Benchmarking failed"
            exit 1
        fi
        bash run.sh ${SIZE} ${OP};
    done
done
