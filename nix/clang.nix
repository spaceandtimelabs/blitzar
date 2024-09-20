{ pkgs }:
with pkgs;
let
  gccForLibs = gcc.cc;
in
stdenvNoCC.mkDerivation {
  name = "clang";
  src = pkgs.fetchFromGitHub {
    owner = "llvm";
    repo = "llvm-project";
    rev = "aa91d90";
    hash = "sha256-+UGCC3OEwGpAz1/ZPKaemZvP4Do7+POqZfduP78WtRc=";
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
    lzma
  ];
  NIX_LDFLAGS = "-L${gccForLibs}/lib/gcc/${targetPlatform.config}/${gccForLibs.version} -L${gcc.libc}/lib";
  CFLAGS = "-B${gccForLibs}/lib/gcc/${targetPlatform.config}/${gccForLibs.version} -B${gcc.libc}/lib";
  patches = [
    ./clang_driver.patch

    # Patch compiler_rt so that only the static library versions of the sanitizers are build.
    # This is a workaround to https://github.com/llvm/llvm-project/issues/69056#issuecomment-1781423887.
    # ./compiler_rt.patch
  ];
  postPatch = ''
    substituteInPlace clang/lib/Driver/ToolChains/Gnu.cpp \
      --replace 'GLIBC_PATH_ABC123' '${gcc.libc}/lib'
    substituteInPlace clang/lib/Driver/ToolChains/Gnu.cpp \
      --replace 'GCCLIB_PATH_ABC123' '${gccForLibs}/lib/gcc/${targetPlatform.config}/${gccForLibs.version}'
  '';
  configurePhase = pkgs.lib.strings.concatStringsSep " " [
    "mkdir build; cd build;"
    "cmake"
    "-G \"Ninja\""
    "-DC_INCLUDE_DIRS=${gcc.libc.dev}/include"
    "-DLLVM_TARGETS_TO_BUILD=\"host;NVPTX\""
    "-DLLVM_BUILTIN_TARGETS=\"x86_64-unknown-linux-gnu\""
    "-DLLVM_RUNTIME_TARGETS=\"x86_64-unknown-linux-gnu\""
    "-DLLVM_ENABLE_PROJECTS=\"clang;clang-tools-extra\""

    # clang
    "-DCLANG_DEFAULT_CXX_STDLIB=libc++"

    "-DLLVM_ENABLE_RUNTIMES=\"libcxx;libcxxabi;libunwind;compiler-rt\""
    #"-DLLVM_ENABLE_RUNTIMES=\"compiler-rt\""
    "-DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_CMAKE_BUILD_TYPE=Release"

    # libcxx
    #"-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXX_ENABLE_SHARED=OFF"
    #"-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXX_ENABLE_STATIC=ON"
    #"-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXX_ENABLE_STATIC_ABI_LIBRARY=ON"
    #"-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXX_STATICALLY_LINK_ABI_IN_STATIC_LIBRARY=ON"

    # libcxxabi
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXXABI_USE_LLVM_UNWINDER=ON"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXXABI_ENABLE_SHARED=ON"
    #"-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXXABI_ENABLE_STATIC=ON"
    #"-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXXABI_ENABLE_STATIC_UNWINDER=ON"
    #"-DRUNTIMES_x86_64-unknown-linux-gnu_LIBCXXABI_STATICALLY_LINK_UNWINDER_IN_STATIC_LIBRARY=ON"

    # libunwind
    #"-DRUNTIMES_x86_64-unknown-linux-gnu_LIBUNWIND_ENABLE_STATIC=ON"
    #"-DRUNTIMES_x86_64-unknown-linux-gnu_LIBUNWIND_ENABLE_SHARED=OFF"

    # compiler-rt
    "-DRUNTIMES_x86_64-unknown-linux-gnu_SANITIZER_ALLOW_CXXABI=OFF"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_COMPILER_RT_INCLUDE_TESTS=OFF"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_COMPILER_RT_USE_LIBCXX=ON"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_COMPILER_RT_CXX_LIBRARY=libcxx"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_COMPILER_RT_USE_LLVM_UNWINDER=ON"
    #"-DRUNTIMES_x86_64-unknown-linux-gnu_SANITIZER_CXX_ABI=libcxxabi"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_COMPILER_RT_SANITIZERS_TO_BUILD=\"asan;msan\""
    "-DRUNTIMES_x86_64-unknown-linux-gnu_COMPILER_RT_BUILD_LIBFUZZER=OFF"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_COMPILER_RT_BUILD_XRAY=OFF"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_COMPILER_RT_BUILD_XRAY_NO_PREINIT=OFF"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_COMPILER_RT_BUILD_PROFILE=OFF"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_COMPILER_RT_BUILD_CTX_PROFILE=OFF"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_COMPILER_RT_BUILD_MEMPROF=OFF"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_COMPILER_RT_BUILD_ORC=OFF"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_COMPILER_RT_BUILD_GWP_ASAN=OFF"
    "-DRUNTIMES_x86_64-unknown-linux-gnu_COMPILER_RT_TEST_STANDALONE_BUILD_LIBS=OFF"
    #"-DRUNTIMES_x86_64-unknown-linux-gnu_COMPILER_RT_SCUDO_STANDALONE_BUILD_SHARED=OFF"

    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_INSTALL_PREFIX=\"$out\""
    "../llvm"
  ];
  buildPhase = "ninja";
  installPhase = "ninja install";
}
