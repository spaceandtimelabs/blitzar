{ pkgs }:
with pkgs;
let
  gccForLibs = gcc13.cc;
in
stdenvNoCC.mkDerivation {
  name = "clang";
  src = pkgs.fetchgit {
    url = "https://github.com/llvm/llvm-project";
    rev = "cb5612c";
    hash = "sha256-7Dyrs+wRYaFbMrt9ioTJxbfaFTzCjG4QMhiPjX5PnaA=";
  };
  nativeBuildInputs = [
    cmake
    perl
    ninja
    python3
  ];
  buildInputs = [
    gcc13
  ];
  NIX_LDFLAGS = "-L${gccForLibs}/lib/gcc/${targetPlatform.config}/${gccForLibs.version} -L${gcc13.libc}/lib";
  CFLAGS = "-B${gccForLibs}/lib/gcc/${targetPlatform.config}/${gccForLibs.version} -B${gcc13.libc}/lib";
  patches = [
    ./clang_driver.patch
  ];
  postPatch = ''
    substituteInPlace clang/lib/Driver/ToolChains/Gnu.cpp \
      --replace 'GLIBC_PATH_ABC123' '${gcc13.libc}/lib'
  '';
  configurePhase = pkgs.lib.strings.concatStringsSep " " [
    "mkdir build; cd build;"
    "cmake"
    "-G \"Unix Makefiles\""
    "-DGCC_INSTALL_PREFIX=${gccForLibs}"
    "-DC_INCLUDE_DIRS=${gcc13.libc.dev}/include"
    "-DLLVM_TARGETS_TO_BUILD=\"host;NVPTX\""
    "-DLLVM_BUILTIN_TARGETS=\"x86_64-unknown-linux-gnu\""
    "-DLLVM_RUNTIME_TARGETS=\"x86_64-unknown-linux-gnu\""
    "-DLLVM_ENABLE_PROJECTS=\"clang;clang-tools-extra\""
    "-DLLVM_ENABLE_RUNTIMES=\"libcxx;libcxxabi;libunwind;compiler-rt\""
    "-DRUNTIMES_x86_64-unknown-linux-gnu_CMAKE_BUILD_TYPE=Release"

    # libcxx
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXX_ENABLE_SHARED=ON"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXX_ENABLE_STATIC=ON"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXX_ENABLE_STATIC_ABI_LIBRARY=ON"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXX_STATICALLY_LINK_ABI_IN_STATIC_LIBRARY=ON"

    # libcxxabi
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXXABI_ENABLE_STATIC=ON"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXXABI_ENABLE_STATIC_UNWINDER=ON"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXXABI_STATICALLY_LINK_UNWINDER_IN_STATIC_LIBRARY=ON"

    # libunwind
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBUNWIND_ENABLE_STATIC=ON"

    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_INSTALL_PREFIX=\"$out\""
    "../llvm"
  ];
  buildPhase = "make";
  installPhase = "make install";
}
