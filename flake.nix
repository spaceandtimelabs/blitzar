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
          };

          config = {
            extraCommands = 
              # taken from https://github.com/nix-community/docker-nixpkgs/blob/master/images/devcontainer/default.nix
              # # create the Nix DB
              # export NIX_REMOTE=local?root=$PWD
              # export USER=nobody
              # ${nix}/bin/nix-store --load-db < ${closureInfo { rootPaths = [ profile ]; }}/registration

              # # set the user profile
              # ${profile}/bin/nix-env --profile nix/var/nix/profiles/default --set ${profile}

              ''
              # minimal
              mkdir -p bin /usr/bin
              ln -s /nix/var/nix/profiles/default/bin/sh bin/sh
              ln -s /nix/var/nix/profiles/default/bin/sh bin/bash
              ln -s /nix/var/nix/profiles/default/bin/env usr/bin/env

              # make sure /tmp exists
              mkdir -m 0777 tmp
            '';
            Cmd = [
              "/bin/bash"
            ];
          };
        };
    };
}
