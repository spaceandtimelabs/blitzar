{ pkgs, clang, cuda }:
let
  # Set PATH so that it only includes things bazel needs.
  path = pkgs.lib.strings.concatStringsSep ":" [
    "${pkgs.git}/bin"
    "${pkgs.gcc}/bin"
    "${pkgs.gcc.libc.bin}/bin"
    "${pkgs.binutils}/bin"
    "${pkgs.coreutils}/bin"
    "${pkgs.findutils}/bin"
    "${pkgs.gnused}/bin"
    "${pkgs.gnugrep}/bin"
    "${pkgs.bash}/bin"
    "${cuda}/bin"
  ];
  bazel = "${pkgs.bazel_7}/bin/bazel";
in
pkgs.writeShellScriptBin "bazel" ''
  if [[
    "$1" == "build" ||
    "$1" == "test" ||
    "$1" == "run"
  ]]; then
    # unset TMPDIR TMP
    exec ${bazel} $1 \
     --action_env PATH="${path}" \
     --action_env CUDA_PATH="${cuda}" \
     ''${@:2}
  else
    exec ${bazel} $@
  fi''

