#!/bin/sh

set -e

# Hack to get around bazel's inability to specify paths relative to the workspace directory
#
# See https://stackoverflow.com/a/74297943
bazel_dir=$(dirname -- "$(readlink -f $0;)")/
ln -sf ${bazel_dir}/lsan.supp /tmp/sxt-blitzar-lsan.supp
