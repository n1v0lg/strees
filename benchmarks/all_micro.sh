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
    64
    128
    256
    1024
)

PID=-1
if [ "$#" -eq 1 ]
then
    echo "Running benchmarks in networked mode"
    PID=$1
else
    echo "Running benchmarks in local mode"
fi

for OP in "${OPS[@]}";
do
    for SIZE in "${SIZES[@]}";
    do
        if [ "$?" -ne 0 ]; then
            echo "Benchmarking failed"
            exit 1
        fi
        bash run.sh micro.py "${OP}-${SIZE}" ${PID};
    done
done
