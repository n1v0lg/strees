#!/bin/bash

HERE=$(cd `dirname $0`; pwd)
SPDZROOT=/home/nikolaj/Desktop/work/MP-SPDZ
cd ${SPDZROOT}

# .py extension for sublime-formatter
./compile.py --ring=64  ${HERE}/strees.py 
./Scripts/ring.sh strees.py
