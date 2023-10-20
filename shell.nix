{ pkgs ? import <nixpkgs> { system = "x86_64-linux"; } }:
with pkgs;
mkShell {
  buildInputs = [
    pkgs.python3
    pkgs.bazel_6
  ];
  shellHook = ''
  '';
}
# (pkgs.buildFHSEnvChroot {
#   name = "simple-env";
#   targetPkgs = pkgs: [
#     pkgs.python3
#     pkgs.gcc13.libc
#     pkgs.bazel_6
#   ];
#   runScript = "bash";
# }).env
