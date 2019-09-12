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
    PORT=8042
    HOST="MP-SPDZ-0"
    RUN_OPTS="${PID} -pn ${PORT} -h ${HOST} -u"
    ./replicated-ring-party.x ${RUN_OPTS} ${MPC_PROG_NAME}
}

function debug_mode() {
    if [ "$LOCAL" = true ]
    then
        compile && run_program_local
    else
        compile && run_program_networked
    fi

    if [ "$?" -ne 0 ]; then
        echo "Debug run failed"
        exit 1
    fi
}

function benchmark_mode() {
    echo "Benchmarking ${MPC_SRC_NAME} with ${PROG_ARGS}"

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
    if [ "$LOCAL" = true ]
    then
        { time run_program_local; } > ${TMP_OUT} 2>${TMP_ERR}
    else
        { time run_program_networked; } > ${TMP_OUT} 2>${TMP_ERR}
    fi

    EXEC_TIME=$(parse_exec_time "$TMP_OUT" "$TMP_ERR")

    # output results
    echo ${MPC_SRC_NAME},${PROG_ARGS},${COMP_TIME},${EXEC_TIME} >> ${HERE}/${OUT_NAME}

    # clean up temp files
    rm "$TMP_OUT" "$TMP_ERR"
}

# configurable parameters
# TODO make argument handling more robust
MPC_SRC_NAME=$1
PROG_ARGS=$2
PID=$3

LOCAL=true
if [ "$PID" -ne -1 ]; then
    echo "Running benchmark in networked mode"
    LOCAL=false
fi

DEBUG=false
if [ "$#" -eq 4 ]; then
    echo "Running benchmark in debug mode"
    DEBUG=true
fi

# fixed parameters
PAR_OPENS=1000
UNROLLING=100000
COMPILE_OPTS="--ring=64 --optimize-hard --insecure -m ${PAR_OPENS} --budget=${UNROLLING}"
MPC_PROG_NAME="${MPC_SRC_NAME}-${PROG_ARGS}"
OUT_NAME="timing-${MPC_SRC_NAME}.csv"

HERE=$(cd `dirname $0`; pwd)
# TODO hacky
SPDZ_ROOT=${HERE}/../../MP-SPDZ
cd ${SPDZ_ROOT}

if [ "$DEBUG" = true ]
then
    debug_mode
else
    benchmark_mode
fi
