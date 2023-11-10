{ pkgs }:
let
  clang = import ./clang.nix { inherit pkgs; };
  compiler-rt = import ./compiler_rt.nix { inherit pkgs; inherit clang; };
  cuda = import ./cuda.nix { inherit pkgs; };
  bazel = import ./bazel.nix { inherit pkgs; inherit clang; inherit cuda; };
in
with pkgs;
mkShell {
  buildInputs = [
    pkgs.bazel-buildtools
    pkgs.python3
    bazel
    clang
    # compiler-rt
    cuda
  ];
  LD_LIBRARY_PATH = lib.makeLibraryPath [
    "/usr/lib/wsl"
    linuxPackages.nvidia_x11
  ];
  shellHook = ''
  '';
}
