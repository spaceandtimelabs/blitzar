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
    in
    {
      formatter.${system} = pkgs.nixpkgs-fmt;
      devShells.${system}.default = import ./nix/shell.nix {
        inherit pkgs;
      };
    };
}
