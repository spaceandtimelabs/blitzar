{ pkgs, clang }:
let
  # Set PATH so that it only includes things bazel needs.
  path = pkgs.lib.strings.concatStringsSep ":" [
    "${clang}/bin"
    "${pkgs.git}/bin"
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
     --copt=-stdlib=libc++ \
     --cxxopt=-stdlib=libc++ \
     --linkopt=-stdlib=libc++ \
     --linkopt=-static-libstdc++ \
     --@rules_cuda//cuda:copts=-stdlib=libc++ \
     --@rules_cuda//cuda:host_copts=-stdlib=libc++ \
     --action_env CC=${clang}/bin/clang \
     --action_env CXX=${clang}/bin/clang++ \
     --action_env PATH="${path}:/nix/store/18bs92p6yf6w2wwxhbplgx02y6anq092-gcc-wrapper-12.3.0/bin" \
     --action_env=BAZEL_LINKLIBS='-l%:libc++.a' \
     --action_env=BAZEL_LINKOPTS='-L${clang}/lib -static-libstdc++' \
     ''${@:2}
  else
    exec ${bazel} $@
  fi''

