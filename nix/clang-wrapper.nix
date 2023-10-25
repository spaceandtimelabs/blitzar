{ pkgs, clang }:
let
  bindir = ${clang}/bin
in
pkgs.writeShellScriptBin "clangWrapper" ''
    exec $bindir/$0 $@
''
