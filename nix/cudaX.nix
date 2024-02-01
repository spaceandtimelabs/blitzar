{ pkgs }:
with pkgs;
pkgs.stdenvNoCC.mkDerivation {
  name = "cudaXWrapped";
  src = fetchurl {
    url = "https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run";
    sha256 = "24b2afc9f770d8cf43d6fa7adc2ebfd47c4084db01bdda1ce3ce0a4d493ba65b";
  };
  unpackPhase = ''
    sh $src --keep --noexec
  '';
  installPhase = ''
    mkdir -p $out/bin $out/lib64 $out/include
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
  '';
}
