#!/bin/bash

set -e

BAZELISK_VERSION=v1.1.0

wget -O /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/$BAZELISK_VERSION/bazelisk-linux-amd64
chmod +x /usr/local/bin/bazel
