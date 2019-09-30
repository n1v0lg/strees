#!/usr/bin/env bash

# Operations to benchmark
declare -a OPS=(
    "compute_ginis"
    "prep" # offline prep, i.e., sorting and encoding sort net as perm
    "dummy_perm_sort" # sorting via permutation network approach
    "dummy_sort_sort" # sorting via pre-computed sorting network approach
    "single_perm_dummy" # single iteration of c45, without pre-processing, using permutation network
    "single_sort_dummy" # single iteration of c45, without pre-processing, using sorting network
#    "single_perm_both" # pre-processing + single iteration of c45, using permutation network
)

# Number continuous attributes
declare -a CONT_ATTRS=(
    2
)

# Number of samples
declare -a SIZES=(
    512
    1024
    2048
    4096
    8192
)

PID=-1
MODE=both
DEBUG_STR=""
MAL_STR=""
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
    --mal)
    MAL_STR="--mal"
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
            ARG="${OP}-${SIZE}-${CONT_ATTR}"
            bash run.sh --source breakdown.py --args ${ARG} --mode ${MODE} --pid ${PID} ${DEBUG_STR} ${MAL_STR}
        done
    done
done
