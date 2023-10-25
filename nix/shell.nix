{ pkgs }:
let
  bazel = import ./bazel.nix { inherit pkgs; };
  clang = import ./clang.nix { inherit pkgs; };
  cuda = import ./cuda.nix { inherit pkgs; };
in
with pkgs;
mkShell {
  buildInputs = [
    pkgs.python3
    bazel
    clang
    cuda
  ];
  shellHook = ''
  '';
}
