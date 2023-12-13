#!/bin/bash
set -eou pipefail

DST_SO_LIB_PATH=$1
NEW_VERSION=$2

# Validate the new version
if ! [[ ${NEW_VERSION} =~ ^[0-9]+[.][0-9]+[.][0-9]+$ ]]
then
    echo "Incorrect semantic version format: " $NEW_VERSION
    exit 1
fi

INCLUDE_FILE="blitzar_api"
INCLUDE_PATH="cbindings"
LIB_PATH="blitzar-sys"
RUST_PATH="$(pwd)/rust"
SRC_SO_LIB_PATH="bazel-bin/cbindings/libblitzar.so"

# Generate the rust bindings based on the C bindings
bindgen --allowlist-file ${INCLUDE_PATH}/${INCLUDE_FILE}.h ${INCLUDE_PATH}/${INCLUDE_FILE}.h -o ${RUST_PATH}/${LIB_PATH}/src/bindings.rs

# Build the Shared Library
bazel build -c opt --config=portable_glibc //cbindings:libblitzar.so

# Copy the Shared Library to the `DST_SO_LIB_PATH
if ! cmp -s $SRC_SO_LIB_PATH $DST_SO_LIB_PATH; then
    cp $SRC_SO_LIB_PATH $DST_SO_LIB_PATH
fi

# Update the version in the blitzar-sys/Cargo.toml
sed -i 's/version = "*.*.*" # DO NOT CHANGE/version = "'${NEW_VERSION}'" # DO NOT CHANGE/' ${RUST_PATH}/${LIB_PATH}/Cargo.toml

# Generate the release assets and publish the crate to crates.io
if [ "$#" -eq 3 ]; then
    if [[ $3 == "--with-release" ]]
    then
        DIST_PATH="$(pwd)/dist"

        mkdir -p ${DIST_PATH}
        cp -f $DST_SO_LIB_PATH ${DIST_PATH}/$(basename "$DST_SO_LIB_PATH")
        cp -f ${INCLUDE_PATH}/${INCLUDE_FILE}.h ${DIST_PATH}/${INCLUDE_FILE}.h

        cd ${RUST_PATH}/${LIB_PATH}
        cargo clean
        cd ..
        zip -r ${DIST_PATH}/blitzar-sys-v${NEW_VERSION}.zip ${LIB_PATH}
        tar -czvf ${DIST_PATH}/blitzar-sys-v${NEW_VERSION}.tar.gz ${LIB_PATH}

        cd ${LIB_PATH}
        cargo publish --allow-dirty --token ${CRATES_TOKEN}
    fi
fi
