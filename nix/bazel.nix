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
     --action_env CXX=clang++ \
     --action_env=BAZEL_LINKLIBS='-l%:libc++.a' \
     --action_env=BAZEL_LINKOPTS=-static-libstdc++ \
     ''${@:2}
  else
    exec ${bazel} $@
  fi''

