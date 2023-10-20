{ pkgs ? import <nixpkgs> { } }:
(pkgs.buildFHSEnvChroot {
  name = "simple-env";
  targetPkgs = pkgs: [
    pkgs.python3
    pkgs.glibc
    pkgs.gcc13.libc
    pkgs.gcc13
    pkgs.clang
    pkgs.bazel_6
  ];
  runScript = "bash";
}).env
