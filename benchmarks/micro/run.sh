#!/usr/bin/env bash

parse_compile_time() {
    # TODO this is hacky
    echo "$1" | grep "^real"
}

parse_exec_time() {
    # TODO this is hacky
    echo "$1" | grep "^real"
    echo "$1" | grep "^Time"
    echo "$1" | grep "^Data sent"
}

MPC_SRC_NAME=run.py
COMPILE_OPTS="--ring=64 --optimize-hard --insecure"

HERE=$(cd `dirname $0`; pwd)
SPDZ_ROOT=${HERE}/../../../MP-SPDZ
cd ${SPDZ_ROOT}

# Format time to only output real in seconds
TIMEFORMAT='real %3R'
parse_compile_time "$({ time ./compile.py ${COMPILE_OPTS} ${HERE}/${MPC_SRC_NAME}; } 2>&1)"
parse_exec_time "$({ time ./Scripts/ring.sh ${MPC_SRC_NAME}; } 2>&1)"
