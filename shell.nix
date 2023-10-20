{ pkgs ? import <nixpkgs> { system = "x86_64-linux"; } }:
(pkgs.buildFHSEnvChroot {
  name = "simple-env";
  targetPkgs = pkgs: [
    pkgs.python3
    pkgs.gcc13.libc
    pkgs.bazel_6
  ];
  runScript = "bash";
}).env
