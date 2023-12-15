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
  ];
  shellHook = ''
    export LIBCLANG_PATH=${llvmPackages.libclang.lib}/lib
    export BINDGEN_EXTRA_CLANG_ARGS="$(< ${stdenv.cc}/nix-support/libc-crt1-cflags) \
      $(< ${stdenv.cc}/nix-support/libc-cflags) \
      $(< ${stdenv.cc}/nix-support/cc-cflags) \
      $(< ${stdenv.cc}/nix-support/libcxx-cxxflags) \
      ${lib.optionalString stdenv.cc.isClang "-idirafter ${stdenv.cc.cc}/lib/clang/${lib.getVersion stdenv.cc.cc}/include"} \
      ${lib.optionalString stdenv.cc.isGNU "-isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc} -isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc}/${stdenv.hostPlatform.config} -idirafter ${stdenv.cc.cc}/lib/gcc/${stdenv.hostPlatform.config}/${lib.getVersion stdenv.cc.cc}/include"} \
    "
  '';
}
