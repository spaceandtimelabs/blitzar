#!/bin/bash

set -e

export DEBIAN_FRONTEND="noninteractive"
export TZ=Etc/UTC

# Install rust
curl https://sh.rustup.rs -sSf | bash -s -- -y --profile minimal

echo 'source $HOME/.cargo/env' >> $HOME/.bashrc

source "$HOME/.cargo/env"

rustup install 1.71.1

rustup component add rustfmt

$HOME/.cargo/bin/cargo install --version 0.66.1 bindgen-cli
