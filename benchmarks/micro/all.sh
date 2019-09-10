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
        bash run.sh ${SIZE} ${OP};
    done
done
