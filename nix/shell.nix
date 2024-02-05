{ pkgs }:
let
  clang = import ./clang.nix { inherit pkgs; };
  cuda = import ./cuda.nix { inherit pkgs; };
  bazel = import ./bazel.nix { inherit pkgs; inherit clang; inherit cuda; };
in
with pkgs;
  stdenvNoCC.mkDerivation {
  name = "shell";
  buildInputs = [
     gcc13
     bazel-buildtools
     python3
  #   rust-bin.nightly."2023-12-01".default
  #   rust-bindgen
  #   patchelf
  #   nodejs
  #   # custom packages
     bazel
  #   # clang
     cuda
  ];
  LD_LIBRARY_PATH = lib.makeLibraryPath [
    "/usr/lib/wsl"
    cudaDrivers
  ];
  shellHook = ''
  '';
}
