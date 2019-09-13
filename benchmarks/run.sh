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
    if [ "$DO_COMPILE" = true ]
    then
        ./compile.py ${COMPILE_OPTS} ${HERE}/${MPC_SRC_NAME} ${PROG_ARGS}
    fi
}

function run_program_local() {
    if [ "$DO_RUN" = true ]
    then
        ./Scripts/ring.sh ${MPC_PROG_NAME}
    fi
}

function run_program_networked() {
    if [ "$DO_RUN" = true ]
    then
        PORT=8042
        HOST="MP-SPDZ-0"
        RUN_OPTS="${PID} -pn ${PORT} -h ${HOST} -u"
        ./replicated-ring-party.x ${RUN_OPTS} ${MPC_PROG_NAME}
     fi
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

# Command line parsing taken from
# https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
POSITIONAL=()

# Defaults
MODE=both

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --source)
    MPC_SRC_NAME="$2"
    shift # past argument
    shift # past value
    ;;
    --args)
    PROG_ARGS="$2"
    shift # past argument
    shift # past value
    ;;
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
    DEBUG=true
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# -1 for PID indicates that all parties should be run locally
LOCAL=true
if [ "$PID" -ne -1 ]; then
    echo "Running benchmark in networked mode"
    LOCAL=false
fi

# fixed parameters
PAR_OPENS=10000
UNROLLING=100000
COMPILE_OPTS="--ring=64 --optimize-hard --insecure -m ${PAR_OPENS} --budget=${UNROLLING}"
MPC_PROG_NAME="${MPC_SRC_NAME}-${PROG_ARGS}"
OUT_NAME="timing-${MODE}-${MPC_SRC_NAME}.csv"

DO_COMPILE=false
DO_RUN=false
if [[ (${MODE} = "compile" || ${MODE} = "both") ]]
then
    DO_COMPILE=true
fi

if [[ (${MODE} = "run" || ${MODE} = "both") ]]
then
    DO_RUN=true
fi

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
