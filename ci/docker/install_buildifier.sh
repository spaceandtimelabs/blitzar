#!/bin/bash

set -e

export DEBIAN_FRONTEND="noninteractive"
export TZ=Etc/UTC

# Install Go
wget https://go.dev/dl/go1.18.4.linux-amd64.tar.gz
tar -xzvf go1.18.4.linux-amd64.tar.gz -C /usr/local
echo "export GOPATH=\"/usr/local/go\"" >> /etc/profile
echo "export PATH=\$PATH:/usr/local/go/bin" >> /etc/profile

# Install buildifier
echo "export BUILDIFIER_BIN=\"\$GOPATH/bin/buildifier\"" >> /etc/profile
source /etc/profile

go install github.com/bazelbuild/buildtools/buildifier@5.1.0
