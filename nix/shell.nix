{ pkgs }:
let
  clang = import ./clang.nix { inherit pkgs; };
  clangWrap = import ./clang-wrapper.nix { inherit pkgs; inherit clang; };
  bazel = import ./bazel.nix { inherit pkgs; inherit clang; };
  cuda = import ./cuda.nix { inherit pkgs; };
in
with pkgs;
mkShell {
  buildInputs = [
    pkgs.python3
    bazel
    clangWrap.clang
    clangWrap.clangpp
    cuda
  ];
  LD_LIBRARY_PATH = lib.makeLibraryPath [
    "/usr/lib/wsl"
    linuxPackages.nvidia_x11
  ];
  shellHook = ''
  '';
}
