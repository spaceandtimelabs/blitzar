{ pkgs }:
with pkgs;
let
  gcc = portableGcc;
  gccForLibs = gcc.cc;
in
stdenvNoCC.mkDerivation {
  name = "clang";
  src = pkgs.fetchFromGitHub {
    owner = "llvm";
    repo = "llvm-project";
    rev = "2405253";
    hash = "sha256-DG69bHVWqCn09CUcx3uglRp7H0LrBED36NY2TNc0yzM=";
  };
  nativeBuildInputs = [
    cmake
    perl
    ninja
    python3
    git
  ];
  buildInputs = [
    gcc
  ];
  NIX_LDFLAGS = "-L${gccForLibs}/lib/gcc/${targetPlatform.config}/${gccForLibs.version} -L${gcc.libc}/lib";
  CFLAGS = "-B${gccForLibs}/lib/gcc/${targetPlatform.config}/${gccForLibs.version} -B${gcc.libc}/lib";
  patches = [
    ./clang_driver.patch

    # Patch compiler_rt so that only the static library versions of the sanitizers are build.
    # This is a workaround to https://github.com/llvm/llvm-project/issues/69056#issuecomment-1781423887.
    ./compiler_rt.patch
  ];
  postPatch = ''
    substituteInPlace clang/lib/Driver/ToolChains/Gnu.cpp \
      --replace 'GLIBC_PATH_ABC123' '${gcc.libc}/lib'
  '';
  configurePhase = pkgs.lib.strings.concatStringsSep " " [
    "mkdir build; cd build;"
    "cmake"
    "-G \"Ninja\""
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
    # "-DGCC_INSTALL_PREFIX=${gccForLibs}"
    "-DC_INCLUDE_DIRS=${gcc.libc.dev}/include"
    "-DLLVM_TARGETS_TO_BUILD=\"host;NVPTX\""
    "-DLLVM_BUILTIN_TARGETS=\"x86_64-unknown-linux-gnu\""
    "-DLLVM_RUNTIME_TARGETS=\"x86_64-unknown-linux-gnu\""
    "-DLLVM_ENABLE_PROJECTS=\"clang;clang-tools-extra\""

    # clang
    "-DCLANG_DEFAULT_CXX_STDLIB=libc++"

    # "-DLLVM_ENABLE_RUNTIMES=\"libcxx;libcxxabi;libunwind;compiler-rt\""
    "-DLLVM_ENABLE_RUNTIMES=\"libcxx;libcxxabi;libunwind\""
    "-DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_CMAKE_BUILD_TYPE=Release"

    # libcxx
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXX_ADDITIONAL_COMPILER_FLAGS=-fPIC"
    #"-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXX_ENABLE_SHARED=OFF"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXX_ENABLE_SHARED=ON"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXX_ENABLE_STATIC=ON"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXX_ENABLE_STATIC_ABI_LIBRARY=ON"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXX_STATICALLY_LINK_ABI_IN_STATIC_LIBRARY=ON"

    # libcxxabi
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXXABI_ADDITIONAL_COMPILER_FLAGS=-fPIC"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXXABI_USE_LLVM_UNWINDER=ON"
    #"-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXXABI_ENABLE_SHARED=OFF"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXXABI_ENABLE_STATIC=ON"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXXABI_ENABLE_STATIC_UNWINDER=ON"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXXABI_STATICALLY_LINK_UNWINDER_IN_STATIC_LIBRARY=ON"

    # libunwind
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBUNWIND_ADDITIONAL_COMPILER_FLAGS=-fPIC"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBUNWIND_ENABLE_STATIC=ON"
    #"-DRUNTIMES_x86_64-unknown-linux-gnu_LIBUNWIND_ENABLE_SHARED=OFF"

    # compiler-rt
    #"-DRUNTIMES_x86_64-unknown-linux-gnu_COMPILER_RT_CXX_LIBRARY=libcxx"
    #"-DRUNTIMES_x86_64-unknown-linux-gnu_COMPILER_RT_USE_LLVM_UNWINDER=ON"
    #"-DRUNTIMES_x86_64-unknown-linux-gnu_COMPILER_RT_SCUDO_STANDALONE_BUILD_SHARED=OFF"

    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_INSTALL_PREFIX=\"$out\""
    "../llvm"
  ];
  buildPhase = "ninja -j4";
  installPhase = "ninja install";
}
