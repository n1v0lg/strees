#!/usr/bin/env bash

# this is a pretty clunky test setup...
ERROR_FLAG="MPC_ERROR"
if bash run.sh | grep ${ERROR_FLAG}; then
    echo "Some tests failed."
    exit 1
else
    echo "All tests OK."
fi
