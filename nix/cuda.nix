# Set up a custom install for cuda toolkit
#
# This is derived from
#  https://github.com/NixOS/nixpkgs/blob/72db73af3db9392554b89eb9aeb3187ae9dca78e/pkgs/development/cuda-modules/cudatoolkit/default.nix
{ pkgs }:
with pkgs;
pkgs.stdenvNoCC.mkDerivation {
  name = "cudatoolkit";
  src = fetchurl {
    url = "https://developer.download.nvidia.com/compute/cuda/12.6.1/local_installers/cuda_12.6.1_560.35.03_linux.run";
    sha256 = "179fx44j0f6fm4m8afx5vhivv3ywyv7mv7shb7r2b5ji8drcxb3k";
  };
  patches = [
    # patch host_defines.h to work with libc++
    ./cuda_host_defines.patch
  ];
  unpackPhase = ''
    sh $src --keep --noexec
  '';
  installPhase = ''
    mkdir -p $out/bin $out/lib64 $out/include
    cp -r pkg/builds $out/builds
    for dir in pkg/builds/*; do
      if [ -d $dir/bin ]; then
        mv $dir/bin/* $out/bin
      fi
      if [ -L $dir/include ] || [ -d $dir/include ]; then
        (cd $dir/include && find . -type d -exec mkdir -p $out/include/\{} \;)
        (cd $dir/include && find . \( -type f -o -type l \) -exec mv \{} $out/include/\{} \;)
      fi
      if [ -L $dir/lib64 ] || [ -d $dir/lib64 ]; then
        (cd $dir/lib64 && find . -type d -exec mkdir -p $out/lib64/\{} \;)
        (cd $dir/lib64 && find . \( -type f -o -type l \) -exec mv \{} $out/lib64/\{} \;)
      fi

    done
    mv pkg/builds/cuda_nvcc/nvvm $out/nvvm
    mv pkg/builds/cuda_sanitizer_api $out/cuda_sanitizer_api
    ln -s $out/cuda_sanitizer_api/compute-sanitizer/compute-sanitizer $out/bin/compute-sanitizer

    mv pkg/builds/nsight_systems/target-linux-x64 $out/target-linux-x64
    mv pkg/builds/nsight_systems/host-linux-x64 $out/host-linux-x64
    rm $out/host-linux-x64/libstdc++.so*
  '';
}
