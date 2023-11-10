{ pkgs, clang }:
with pkgs;
let
  gccForLibs = gcc13.cc;
in
stdenvNoCC.mkDerivation {
  name = "compiler-rt";
  src = pkgs.fetchgit {
    url = "https://github.com/llvm/llvm-project";
    rev = "a396fb2";
    hash = "sha256-BUgfgs46LqrwZy3/vQbw9vgH2dTVlguaxoFOAqATadI=";
  };
  nativeBuildInputs = [
    cmake
    perl
    ninja
    python3
    git
  ];
  buildInputs = [
    clang
    gcc13
  ];
  configurePhase = pkgs.lib.strings.concatStringsSep " " [
    "mkdir build; cd build;"
    "cmake"
    "-G \"Unix Makefiles\""
    "-DLLVM_CMAKE_DIR=$src/cmake/modules"
    # "-DGCC_INSTALL_PREFIX=${gccForLibs}"
    # "-DC_INCLUDE_DIRS=${gcc13.libc.dev}/include"
    # "-DCMAKE_C_COMPILER=${clang}/bin/clang"
    # "-DCMAKE_CPP_COMPILER=${clang}/bin/clang++"
    # "-DLLVM_TARGETS_TO_BUILD=\"host;NVPTX\""
    "-DLLVM_BUILTIN_TARGETS=\"x86_64-unknown-linux-gnu\""
    "-DLLVM_RUNTIME_TARGETS=\"x86_64-unknown-linux-gnu\""
    # "-DLLVM_ENABLE_PROJECTS=\"clang;clang-tools-extra\""

    # TODO(rnburn): build with compiler-rt so that we have access to
    # sanitizers after this issue gets resolved: https://github.com/llvm/llvm-project/issues/69056#issuecomment-1781423887.
    #"-DLLVM_ENABLE_RUNTIMES=\"libcxx;libcxxabi;libunwind;compiler-rt\""
    # "-DLLVM_ENABLE_RUNTIMES=\"libcxx;libcxxabi;libunwind\""
    # "-DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF"
    # "-DRUNTIMES_x86_64-unknown-linux-gnu_CMAKE_BUILD_TYPE=Release"

    # libcxx
    # "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXX_ENABLE_SHARED=ON"
    # "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXX_ENABLE_STATIC=ON"
    # "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXX_ENABLE_STATIC_ABI_LIBRARY=ON"
    # "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXX_STATICALLY_LINK_ABI_IN_STATIC_LIBRARY=ON"

    # libcxxabi
    # "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXXABI_USE_LLVM_UNWINDER=ON"
    # "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXXABI_ENABLE_STATIC=ON"
    # "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXXABI_ENABLE_STATIC_UNWINDER=ON"
    # "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXXABI_STATICALLY_LINK_UNWINDER_IN_STATIC_LIBRARY=ON"

    # libunwind
    # "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBUNWIND_ENABLE_STATIC=ON"

    # compiler-rt
    # "-DCOMPILER_RT_USE_LLVM_UNWINDER=ON"
    # "-DCOMPILER_CXX_LIBRARY=libcxx"
    # "-DCOMPILER_RT_USE_BUILTINS_LIBRARY=ON"

    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_INSTALL_PREFIX=\"$out\""
    "../compiler-rt"
  ];
  buildPhase = "make";
  installPhase = "make install";
}
