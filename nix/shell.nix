{ pkgs }:
let
  clang = import ./clang.nix { inherit pkgs; };
  cuda = import ./cuda.nix { inherit pkgs; };
  bazel = import ./bazel.nix { inherit pkgs; inherit clang; inherit cuda; };
in
with pkgs;
mkShell {
  buildInputs = [
    pkgs.bazel-buildtools
    pkgs.python3
    pkgs.cargo
    pkgs.rust-bindgen
    pkgs.rustfmt
    bazel
    clang
    cuda
  ];
  LD_LIBRARY_PATH = lib.makeLibraryPath [
    "/usr/lib/wsl"
    cudaDrivers
  ];
  shellHook = ''
  '';
}
