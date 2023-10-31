{ pkgs }:
let
  clang = import ./clang.nix { inherit pkgs; };
  bazel = import ./bazel.nix { inherit pkgs; inherit clang; };
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
  LD_LIBRARY_PATH = lib.makeLibraryPath [
    "/usr/lib/wsl"
    linuxPackages.nvidia_x11
  ];
  shellHook = ''
  '';
}
