#!/bin/bash

set -e

export DEBIAN_FRONTEND="noninteractive"
export TZ=Etc/UTC

apt-get update
apt-get install --no-install-recommends --no-install-suggests -y \
                software-properties-common \
                build-essential \
                zip \
                git \
                ca-certificates \
                curl \
                gnupg2 \
                ssh \
                wget \
                python3 \
                python3-pip \
                llvm-dev \
                libclang-dev \
                clang \
                graphviz \
                valgrind

# Upgrade to clang-18
# See https://linux.how2shout.com/how-to-install-clang-on-ubuntu-linux/
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
./llvm.sh 18
update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 100
update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 100

# Install benchmark dependencies
pip install gprof2dot===2022.7.29
pip install matplotlib===3.7.0
pip install slack_sdk===3.20.1
pip install psutil===5.9.4
pip install py-cpuinfo===9.0.0
pip install pandas===1.5.3

