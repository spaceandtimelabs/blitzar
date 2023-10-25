{ pkgs, clang }:
let
  bindir = "${clang}/bin/";
in
{
  clang=pkgs.writeShellScriptBin "clang" ''
    exec ${clang}/bin/clang \
      -stdlib=libc++ \
      -isystem ${clang}/include/x86_64-unknown-linux-gnu/c++/v1 \
      -L ${clang}/lib/x86_64-unknown-linux-gnu/ \
      $@
  '';
  clangpp=pkgs.writeShellScriptBin "clang++" ''
    exec ${clang}/bin/clang++ \
      -stdlib=libc++ \
      -isystem ${clang}/include/x86_64-unknown-linux-gnu/c++/v1 \
      -L ${clang}/lib/x86_64-unknown-linux-gnu/ \
      $@
  '';
}
