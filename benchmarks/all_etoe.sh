#!/usr/bin/env bash

# Tree depth
declare -a DEPTHS=(
    1
    2
    3
)

# Number continuous attributes
declare -a CONT_ATTRS=(
    2
)

# Number of samples
declare -a SIZES=(
    1024
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

for SIZE in "${SIZES[@]}";
do
    for DEPTH in "${DEPTHS[@]}";
    do
        for CONT_ATTR in "${CONT_ATTRS[@]}";
        do
            bash run.sh --source etoe.py --args "${SIZE}-${DEPTH}-${CONT_ATTRS}" --mode ${MODE} --pid ${PID} ${DEBUG_STR}
        done
    done
done
