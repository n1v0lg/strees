#!/usr/bin/env bash

# Ops to benchmark
declare -a OPS=(
    "argmax"
    "shuffle"
    "sort"
    "comp_mat"
    "comp_mat_par"
)

# Default benchmark sizes
declare -a SIZES=(
    128
    256
    1024
    2048
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
        bash run.sh --source micro.py --args "${OP}-${SIZE}" --pid ${PID};
    done
done
