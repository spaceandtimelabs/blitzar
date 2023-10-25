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
  buildPhase = "";

  installPhase = ''
    mkdir $out
    for f in `ls -1 ${toolkit}`
    do
      if [[ $f != "lib64" && $f != "lib" ]]; then
        ln -s ${toolkit}/$f $out/$f
      fi
    done
    mkdir $out/lib64
    for f in `ls -1 ${toolkit}/lib`
    do
      ln -s ${toolkit}/lib/$f $out/lib64/$f
    done
    ln -s ${toolkit-lib}/lib/libcudart_static.a $out/lib64/libcudart_static.a
  '';
}
