{ pkgs }:
let
  clang = import ./clang.nix { inherit pkgs; };
  cuda = import ./cuda.nix { inherit pkgs; };
  bazel = import ./bazel.nix { inherit pkgs; inherit clang; inherit cuda; };
in
with pkgs;
mkShell {
  buildInputs = [
    bazel-buildtools
    python3
    rust-bin.nightly."2023-12-01".default
    # cargo
    # rust-bindgen
    # rustfmt
    patchelf
    nodejs
    # custom
    bazel
    clang
    cuda
  ];
  LD_LIBRARY_PATH = lib.makeLibraryPath [
    "/usr/lib/wsl"
    cudaDrivers
    gcc13.libc
  ];
  shellHook = ''
    export LIBCLANG_PATH=${clang}/lib
  '';
}
