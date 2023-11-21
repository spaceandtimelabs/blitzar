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
      packages.${system}.docker = pkgs.dockerTools.buildImage {
        name = "blitzar";
        config = {
          Cmd = [
            "${pkgs.bash}/bin/bash"
          ];
        };
      };
    };
}
