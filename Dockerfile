# This is a docker file for the main MP-SPDZ repo
FROM ubuntu:18.04
RUN apt-get update && apt-get install -y \
 automake \
 build-essential \
 git \
 libboost-dev \
 libboost-thread-dev \
 libsodium-dev \
 libssl-dev \
 libntl-dev \
 libtool \
 m4 \
 python \
 texinfo \
 yasm

WORKDIR /home/MP-SPDZ
ADD . /home/MP-SPDZ
RUN echo "MY_CFLAGS = -DINSECURE" >> CONFIG.mine
RUN make tldr && make rep-ring
RUN mkdir Player-Data 2> /dev/null
RUN Scripts/setup-ssl.sh 3
