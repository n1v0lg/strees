#!/bin/bash

MPC_SRC_NAME=main.py

HERE=$(cd `dirname $0`; pwd)
SPDZ_ROOT=${HERE}/../MP-SPDZ
cd ${SPDZ_ROOT}

# .py extension for sublime-formatter
./compile.py ${HERE}/${MPC_SRC_NAME}
./Scripts/mascot.sh ${MPC_SRC_NAME}
