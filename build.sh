#!/bin/bash

DIST_PATH="dist/"
CRATE_PATH="proofsgpu-sys"

mkdir -p ${DIST_PATH}

bazel build -c opt //binary:libproofs-gpu.so
mkdir -p ${CRATE_PATH}/proofs-gpu-sys/src/proofsgpulib/

NEW_VERSION="$1"

ORIG_LIB_NAME="libproofs-gpu"
ORIG_LIB_PATH="bazel-bin/binary"

LIB_PATH="${CRATE_PATH}/proofs-gpu-sys/src/proofsgpulib"
LIB_NAME="${ORIG_LIB_NAME}-v${NEW_VERSION}.so-linux-$(uname -m)"

cp ${ORIG_LIB_PATH}/${ORIG_LIB_NAME}.so ${DIST_PATH}/${LIB_NAME}
cp ${ORIG_LIB_PATH}/${ORIG_LIB_NAME}.so ${LIB_PATH}/${LIB_NAME}

INCLUDE_FILE="pedersen_capi"
INCLUDE_PATH="sxt/seqcommit/cbindings"
NEW_INCLUDE_NAME="${INCLUDE_FILE}_v${NEW_VERSION}.h"

cp ${INCLUDE_PATH}/${INCLUDE_FILE}.h ${DIST_PATH}/${NEW_INCLUDE_NAME}
cp ${INCLUDE_PATH}/${INCLUDE_FILE}.h ${LIB_PATH}/${NEW_INCLUDE_NAME}

sed -i 's/version = "*.*.*"/version = "'${NEW_VERSION}'"/' ${CRATE_PATH}/proofs-gpu-sys/Cargo.toml

zip -r ${DIST_PATH}/proofsgpu-sys-v${NEW_VERSION}.zip ${CRATE_PATH}
tar -czvf ${DIST_PATH}/proofsgpu-sys-v${NEW_VERSION}.tar.gz ${CRATE_PATH}