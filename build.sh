#!/bin/bash

HERE=$(cd `dirname $0`; pwd)
SPDZ_ROOT=${HERE}/../MP-SPDZ
cp ./Dockerfile ${SPDZ_ROOT}/Dockerfile
cd ${SPDZ_ROOT}

docker build -t "mpspdz" .
rm ${SPDZ_ROOT}/Dockerfile
docker run -P -ti -v ${HERE}/:/home/strees --name mpspdz mpspdz
