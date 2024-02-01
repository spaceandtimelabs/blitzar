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
    mkdir $out
    echo arf > $out/cat
  '';
}
