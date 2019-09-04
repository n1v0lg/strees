#!/bin/bash

MPC_SRC_NAME=c45.py

HERE=$(cd `dirname $0`; pwd)
SPDZ_ROOT=${HERE}/../MP-SPDZ
cd ${SPDZ_ROOT}

# .py extension so we can use an IDE
# TODO merging open instructions causes c45 test to produce incorrect results
./compile.py --insecure --nomerge ${HERE}/c45/${MPC_SRC_NAME} && ./Scripts/mascot.sh ${MPC_SRC_NAME}
