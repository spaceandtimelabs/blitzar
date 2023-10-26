{ pkgs, clang, clangpp }:
let
  # Set PATH so that it only includes things bazel needs.
  path = pkgs.lib.strings.concatStringsSep ":" [
    "${clang}/bin"
    "${clangpp}/bin"
    "${pkgs.gcc13}/bin"
    "${pkgs.gcc13.libc.bin}/bin"
    "${pkgs.binutils}/bin"
    "${pkgs.coreutils}/bin"
    "${pkgs.findutils}/bin"
    "${pkgs.gnused}/bin"
    "${pkgs.gnugrep}/bin"
    "${pkgs.bash}/bin"
  ];
  bazel = "${pkgs.bazel_6}/bin/bazel";
in
pkgs.writeShellScriptBin "bazel" ''
  if [[
    "$1" == "build" ||
    "$1" == "test" ||
    "$1" == "run"
  ]]; then
    exec ${bazel} $1 \
     --action_env CC=${clang} \
     --action_env CXX=${clangpp} \
     --action_env PATH="${path}" \
     --action_env=BAZEL_LINKLIBS='-l%:libc++.a' \
     --action_env=BAZEL_LINKOPTS=-static-libstdc++ \
     ''${@:2}
  else
    exec ${bazel} $@
  fi''

