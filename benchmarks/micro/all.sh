#!/usr/bin/env bash

# Default benchmark sizes
declare -a SIZES=(
  16
  32
  64
  128
  1024
)

for SIZE in "${SIZES[@]}";
do
    bash run.sh ${SIZE} shuffle;
done

