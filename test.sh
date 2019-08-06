#!/usr/bin/env bash

# this is a pretty clunky test setup...
ERROR_FLAG="MPC_ERROR"
bash run.sh | grep ${ERROR_FLAG} &> /dev/null
if [ $? == 0 ]; then
    echo "Some tests failed."
    exit 1
else
    echo "All tests OK."
fi
