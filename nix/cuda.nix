{ pkgs }:
with pkgs;
let
  toolkit = cudaPackages_12_3.cudatoolkit;
  toolkit-lib = cudaPackages_12_3.cudatoolkit.lib;
in
pkgs.stdenvNoCC.mkDerivation {
  name = "cudaWrapped";

  buildInputs = [
    toolkit
    toolkit-lib
  ];

  src = [
    ./host_defines.h
  ];

  unpackPhase = "true";
  buildPhase = "";

  installPhase = 
  let
    hostDefines = src."./host_defines.h";
  in
  ''
    mkdir $out
    for f in `ls -1 ${toolkit}`
    do
      if [[ $f != "lib64" && $f != "include" && $f != "lib" ]]; then
        ln -s ${toolkit}/$f $out/$f
      fi
    done

    mkdir $out/include
    for f in `ls -1 ${toolkit}/include`
    do
      if [[ $f != "crt" ]]; then
        ln -s ${toolkit}/include/$f $out/include/$f
      fi
    done
    mkdir $out/include/crt
    for f in `ls -1 ${toolkit}/include/crt`
    do
      if [[ $f != "host_defines.h" ]]; then
        ln -s ${toolkit}/include/crt/$f $out/include/crt/$f
      fi
    done

    mkdir $out/lib64
    for f in `ls -1 ${toolkit}/lib`
    do
      ln -s ${toolkit}/lib/$f $out/lib64/$f
    done
    ln -s ${toolkit-lib}/lib/libcudart_static.a $out/lib64/libcudart_static.a
    mkdir $out/patched
    for srcFile in $src; do
      cp $srcFile $out/patched/$(stripHash $srcFile)
    done
    ln -s $out/patched/host_defines.h $out/include/crt/host_defines.h
  '';
}
