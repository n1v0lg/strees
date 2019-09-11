#!/usr/bin/env bash

function parse_compile_time() {
    # TODO this is hacky
    REAL_TIME=$(cat "$1" | grep -oP "^real \K[0-9]+(\.[0-9]+)?")
    echo "$REAL_TIME"
}

function parse_exec_time() {
    # TODO this is hacky
    REAL_TIME=$(cat "$2" | grep -oP "^real \K[0-9]+(\.[0-9]+)?")
    echo "$REAL_TIME"
}

function compile() {
    ./compile.py ${COMPILE_OPTS} ${HERE}/${MPC_SRC_NAME} ${PROG_ARGS}
}

function run_program_local() {
    ./Scripts/ring.sh ${MPC_PROG_NAME}
}

function run_program_networked() {
    replicated-ring-party.x ${RUN_OPTS} ${MPC_PROG_NAME}
}

function debug_mode() {
    compile && run_program_local
    if [ "$?" -ne 0 ]; then
        echo "Debug run failed"
        exit 1
    fi
}

function benchmark_mode() {
    echo "Benchmarking ${OP} on ${NUM_ELS} els"

    # Test-run first, which will exit with error if anything breaks
    # TODO check for errors directly on timed run
    debug_mode > /dev/null 2>&1

    # Format time to only output real in seconds
    TIMEFORMAT='real %3R'

    # Temp files for redirecting stdout and err to separate streams
    TMP_OUT=$(mktemp /tmp/bench.out.XXXXXX)
    TMP_ERR=$(mktemp /tmp/bench.err.XXXXXX)

    { time compile; } > ${TMP_OUT} 2>${TMP_ERR}
    COMP_TIME=$(parse_compile_time "$TMP_ERR")

    # TODO MP-SPDZ appears to exit its main process before finishing all writes
    #  need to figure out a way to wait for all subprocesses to finish before parsing stdout
    { time run_program_local; } > ${TMP_OUT} 2>${TMP_ERR}
    EXEC_TIME=$(parse_exec_time "$TMP_OUT" "$TMP_ERR")

    # output results
    echo ${OP},${NUM_ELS},${COMP_TIME},${EXEC_TIME} >> ${HERE}/${OUT_NAME}

    # clean up temp files
    rm "$TMP_OUT" "$TMP_ERR"
}

# configurable parameters
# TODO make argument handling more robust
NUM_ELS=$1
OP=$2
PID=$3

if [ -z ${NUM_ELS} ]
then
    echo "Supply number of elements for benchmark"
    exit 1
fi

if [ -z ${OP} ]
then
    echo "Supply benchmark type"
    exit 1
fi

LOCAL=true
RUN_OPTS=""
if [ "$PID" -ne -1 ]; then
    echo "Running benchmark in networked mode"
    PORT=8042
    RUN_OPTS="${PID} -pn ${PORT} -h MP-SPDZ-${PID} -u"
    LOCAL=false
    exit 1
fi

DEBUG=false
if [ "$#" -eq 4 ]; then
    echo "Running benchmark in debug mode"
    DEBUG=true
fi

# fixed parameters
MPC_SRC_NAME=bench.py
OUT_NAME=timing.csv
COMPILE_OPTS="--ring=64 --optimize-hard --insecure"
MPC_PROG_NAME="${MPC_SRC_NAME}-${OP}-${NUM_ELS}"
PROG_ARGS="${OP} ${NUM_ELS}"

HERE=$(cd `dirname $0`; pwd)
# TODO hacky
SPDZ_ROOT=${HERE}/../../../MP-SPDZ
cd ${SPDZ_ROOT}

if [ "$DEBUG" = true ]
then
    # TODO own function
    debug_mode
else
    benchmark_mode
fi
