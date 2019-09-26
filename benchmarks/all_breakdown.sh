#!/usr/bin/env bash

# Operations to benchmark
declare -a OPS=(
#    "prep"
#    "dummy_perm_sort"
#    "dummy_sort_sort"
#    "single_perm_dummy"
    "single_perm_both"
#    "single_sort_dummy"
#    "single_sort_both"
)

# Number continuous attributes
declare -a CONT_ATTRS=(
    2
)

# Number of samples
declare -a SIZES=(
#    2048
#    4096
    8192
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
    for CONT_ATTR in "${CONT_ATTRS[@]}";
    do
        for SIZE in "${SIZES[@]}";
        do
            bash run.sh --source breakdown.py --args "${OP}-${SIZE}-${CONT_ATTR}" --mode ${MODE} --pid ${PID} ${DEBUG_STR}
        done
    done
done
