#!/usr/bin/env bash

# Ops to benchmark
declare -a OPS=(
    "argmax"
    "shuffle"
    "sort"
    "comp_mat_par"
    "lt_threshold"
    "is_last_active_lin"
    "is_last_active_log"
)

# Default benchmark sizes
declare -a SIZES=(
    512
    1024
    2048
)

PID=-1
MODE=both
DEBUG_STR=""
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --pid)
    PID="$2"
    shift # past argument
    shift # past value
    ;;
    --mode)
    MODE="$2"
    shift # past argument
    shift # past value
    ;;
    --debug)
    DEBUG_STR="--debug"
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

if [ "$PID" -ne -1 ]
then
    echo "Running benchmarks in networked mode"
else
    echo "Running benchmarks in local mode"
fi



for OP in "${OPS[@]}";
do
    for SIZE in "${SIZES[@]}";
    do
        bash run.sh --source micro.py --args "${OP}-${SIZE}" --mode ${MODE} --pid ${PID} ${DEBUG_STR};
    done
done
