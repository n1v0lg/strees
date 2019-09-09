#!/usr/bin/env bash

parse_compile_time() {
    # TODO this is hacky
    REAL_TIME=$(cat "$1" | grep -E "^real [0-9](\.[0-9]+)?$")
    echo $REAL_TIME
}

parse_exec_time() {
    # TODO this is hacky
    cat "$1" | grep "^Time"
    cat "$1" | grep "^Data sent"
    cat "$2" | grep "^real"
}

main() {
    MPC_SRC_NAME=run.py
    OUT_NAME=timing.csv
    COMPILE_OPTS="--ring=64 --optimize-hard --insecure"

    HERE=$(cd `dirname $0`; pwd)
    # TODO hacky
    SPDZ_ROOT=${HERE}/../../../MP-SPDZ
    cd ${SPDZ_ROOT}

    # Format time to only output real in seconds
    TIMEFORMAT='real %3R'

    # Temp files for redirecting stdout and err to separate streams
    TMP_OUT=$(mktemp /tmp/bench.out.XXXXXX)
    TMP_ERR=$(mktemp /tmp/bench.err.XXXXXX)
    
    { time ./compile.py ${COMPILE_OPTS} ${HERE}/${MPC_SRC_NAME}; } > ${TMP_OUT} 2>${TMP_ERR}
    parse_compile_time "$TMP_ERR"
    { time ./Scripts/ring.sh ${MPC_SRC_NAME}; } > ${TMP_OUT} 2>${TMP_ERR}
    parse_exec_time "$TMP_OUT" "$TMP_ERR"

    rm "$TMP_OUT" "$TMP_ERR"
}

main
