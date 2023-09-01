#!/bin/bash

set -e

IMAGE=spaceandtimelabs/blitzar:12.2.0-cuda-1.71.1-rust-0

# If you have a GPU instance configured in your machine
docker run --rm -v "$PWD":/src -w /src --gpus all --privileged -it "$IMAGE" /bin/bash -l
