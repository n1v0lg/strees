#!/usr/bin/env bash

MPC_SRC_NAME=run.py

HERE=$(cd `dirname $0`; pwd)
SPDZ_ROOT=${HERE}/../../../MP-SPDZ
cd ${SPDZ_ROOT}

(time ./compile.py --ring=64 --optimize-hard --insecure ${HERE}/${MPC_SRC_NAME}) 2>&1 | (grep real)
(time ./Scripts/ring.sh ${MPC_SRC_NAME}) 2>&1 | grep real
