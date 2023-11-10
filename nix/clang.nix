{ pkgs }:
with pkgs;
let
  gccForLibs = gcc13.cc;
in
stdenvNoCC.mkDerivation {
  name = "clang";
  src = pkgs.fetchgit {
    url = "https://github.com/llvm/llvm-project";
    rev = "fdbff88";
    hash = "sha256-kipkrgqzSgdsHwYz5P2NpUo6miulE/Nd9zRgeKAHeHM=";
  };
  nativeBuildInputs = [
    cmake
    perl
    ninja
    python3
    git
  ];
  buildInputs = [
    gcc13
  ];
  NIX_LDFLAGS = "-L${gccForLibs}/lib/gcc/${targetPlatform.config}/${gccForLibs.version} -L${gcc13.libc}/lib";
  CFLAGS = "-B${gccForLibs}/lib/gcc/${targetPlatform.config}/${gccForLibs.version} -B${gcc13.libc}/lib";
  patches = [
    ./clang_driver.patch
    ./compiler_rt.patch
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

    # TODO(rnburn): build with compiler-rt so that we have access to
    # sanitizers after this issue gets resolved: https://github.com/llvm/llvm-project/issues/69056#issuecomment-1781423887.
    "-DLLVM_ENABLE_RUNTIMES=\"libcxx;libcxxabi;libunwind;compiler-rt\""
    # "-DLLVM_ENABLE_RUNTIMES=\"libcxx;libcxxabi;libunwind\""
    "-DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_CMAKE_BUILD_TYPE=Release"

    # libcxx
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXX_ENABLE_SHARED=ON"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXX_ENABLE_STATIC=ON"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXX_ENABLE_STATIC_ABI_LIBRARY=ON"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXX_STATICALLY_LINK_ABI_IN_STATIC_LIBRARY=ON"

    # libcxxabi
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXXABI_USE_LLVM_UNWINDER=ON"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXXABI_ENABLE_STATIC=ON"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXXABI_ENABLE_STATIC_UNWINDER=ON"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXXABI_STATICALLY_LINK_UNWINDER_IN_STATIC_LIBRARY=ON"

    # libunwind
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBUNWIND_ENABLE_STATIC=ON"

    # compiler-rt
    "-DRUNTIMES_x86_64-unknown-linux-gnu_COMPILER_RT_CXX_LIBRARY=libcxx"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_COMPILER_RT_USE_LLVM_UNWINDER=ON"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_COMPILER_RT_SCUDO_STANDALONE_BUILD_SHARED=OFF"

    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_INSTALL_PREFIX=\"$out\""
    "../llvm"
  ];
  buildPhase = "make";
  installPhase = "make install";
}
