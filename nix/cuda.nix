{ pkgs }:
with pkgs;
let
  toolkit = cudaPackages_12_2.cudatoolkit;
  toolkit-lib = cudaPackages_12_2.cudatoolkit.lib;
in
pkgs.stdenvNoCC.mkDerivation {
  name = "cudaWrapped";

  buildInputs = [
    toolkit
    toolkit-lib
  ];

  unpackPhase = "true";

  installPhase = ''
    mkdir -p $out/bin
    mkdir -p $out/lib64
    ln -s ${toolkit}/bin/* $out/bin
    ln -s ${toolkit}/lib/* $out/lib64
    ln -s ${toolkit-lib}/lib/libcudart_static.a $out/lib64/libcudart_static.a
  '';
}
