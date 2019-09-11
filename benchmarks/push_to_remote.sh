#!/usr/bin/env bash

if [ -z "$1" ]
  then
    echo "Need key path"
    exit 1
fi

if [ -z "$2" ]
  then
    echo "Need user name"
    exit 1
fi

KEY_PATH=${1}
USER_NAME=${2}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

declare -a HOSTS=(
    "MP-SPDZ-0"
    "MP-SPDZ-1"
    "MP-SPDZ-2"
)

for HOST in "${HOSTS[@]}";
do
    scp -i ${KEY_PATH} -r ${DIR} ${USER_NAME}@${HOST}:/home/${USER_NAME}/strees/
done
