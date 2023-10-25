{ pkgs, clang }:
let
  bazel = "${pkgs.bazel_6}/bin/bazel";
in
pkgs.writeShellScriptBin "bazel" ''
  if [[
    "$1" == "build" ||
    "$1" == "test" ||
    "$1" == "run"
  ]]; then
    exec ${bazel} $1 \
     --action_env CC=clang \
     --action_env CC=clang++ \
     --copt="-isystem ${clang}/include/x86_64-unknown-linux-gnu/c++/v1" \
     --copt=-stdlib=libc++ \
     --linkopt=-stdlib=libc++ \
     --linkopt="-L ${clang}/lib/x86_64-unknown-linux-gnu" \
     --@rules_cuda//cuda:copts="-isystem ${clang}/include/x86_64-unknown-linux-gnu/c++/v1" \
     --@rules_cuda//cuda:host_copts="-isystem ${clang}/include/x86_64-unknown-linux-gnu/c++/v1" \
     --@rules_cuda//cuda:copts=-stdlib=libc++ \
     --@rules_cuda//cuda:host_copts=copt=-stdlib=libc++ \
     ''${@:2}
  else
    exec ${bazel} $@
  fi''

