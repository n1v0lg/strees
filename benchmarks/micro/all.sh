#!/usr/bin/env bash

# Ops to benchmark
declare -a OPS=(
    "argmax"
    "shuffle"
    "sort"
    "comp_mat"
)

# Default benchmark sizes
declare -a SIZES=(
    8
    16
)

for OP in "${OPS[@]}";
do
    for SIZE in "${SIZES[@]}";
    do
        if [ "$?" -ne 0 ]; then
            echo "Benchmarking failed"
            exit 1
        fi
        bash run.sh ${SIZE} ${OP} -1;
    done
done
