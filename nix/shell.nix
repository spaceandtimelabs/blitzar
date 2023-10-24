{ pkgs }:
let
  clang = import ./clang.nix { inherit pkgs; };
in
with pkgs;
mkShell {
  buildInputs = [
    pkgs.python3
    pkgs.bazel_6
    clang
  ];
  shellHook = ''
  '';
}
