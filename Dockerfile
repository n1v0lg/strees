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
RUN make tldr && make mascot-party.x
RUN mkdir Player-Data 2> /dev/null
