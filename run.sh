#!/bin/bash

MPC_SRC_NAME=$1

HERE=$(cd `dirname $0`; pwd)
SPDZ_ROOT=${HERE}/../MP-SPDZ
cd ${SPDZ_ROOT}

# .py extension so we can use an IDE
./compile.py --ring=64 --optimize-hard --insecure ${HERE}/c45/${MPC_SRC_NAME} && ./Scripts/ring.sh ${MPC_SRC_NAME}
