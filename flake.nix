{
  description = "blitzar build environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { 
        inherit system; 
        config.allowUnfree = true;
        config.cudaSupport = true;
      };
      shell = import ./nix/shell.nix { inherit pkgs; };
    in
    {
      formatter.${system} = pkgs.nixpkgs-fmt;
      devShells.${system}.default = shell;
      packages.${system}.docker = 
        let
          clang = import ./nix/clang.nix { inherit pkgs; };
          cuda = import ./nix/cuda.nix { inherit pkgs; };
          bazel = import ./nix/bazel.nix { inherit pkgs; inherit clang; inherit cuda; };
        in
        with pkgs;
        pkgs.dockerTools.buildImage {
          name = "blitzar";
          copyToRoot = pkgs.buildEnv {
            name = "image-root";
            paths = [
              bazel
              binutils
              coreutils
              bashInteractive
            ];
            pathsToLink = ["/bin"];
          };

          config = {
            Cmd = [
              "/bin/bash"
            ];
          };
        };
    };
}
