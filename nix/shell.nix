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
    llvmPackages.libclang.lib
    gcc13.libc
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
    gcc13.libc
    cudaDrivers
  ];
  shellHook = ''
    export LIBCLANG_PATH=${llvmPackages.libclang.lib}/lib
  '';
}
