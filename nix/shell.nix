{ pkgs }:
let
  clang = import ./clang.nix { inherit pkgs; };
  wrap-clang = import ./clang-wrapper.nix { inherit pkgs; inherit clang; name = "clang"; };
  wrap-clangpp = import ./clang-wrapper.nix { inherit pkgs; inherit clang; name = "clang++"; };
  bazel = import ./bazel.nix { inherit pkgs; inherit clang; };
  cuda = import ./cuda.nix { inherit pkgs; };
in
with pkgs;
mkShell {
  buildInputs = [
    pkgs.python3
    bazel
    wrap-clang
    wrap-clangpp
    clang
    # (wrap-clang "clang")
    # (wrap-clang "clang++")
    cuda
  ];
  LD_LIBRARY_PATH = lib.makeLibraryPath [
    "/usr/lib/wsl"
    linuxPackages.nvidia_x11
  ];
  shellHook = ''
  '';
}
