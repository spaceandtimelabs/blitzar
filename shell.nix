{ pkgs ? import <nixpkgs> { } }:
(pkgs.buildFHSEnv {
  name = "simple-env";
  targetPkgs = pkgs: [
    pkgs.python3
  ];
  runScript = "bash";
}).env
