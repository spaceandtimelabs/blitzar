#!/bin/bash

set -e

# Install latest clang-format
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|apt-key add -
apt-get update
add-apt-repository "deb http://apt.llvm.org/focal/ llvm-toolchain-focal main"
apt-get update
apt-get install --no-install-recommends --no-install-suggests -y \
      clang-format

# Export clang-format
echo "export CLANG_FORMAT=\"$(which clang-format)\"" >> /etc/profile
