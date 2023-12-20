{
  description = "blitzar build environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

    # nixpkgsGcc is pinned to an older version of nixpkgs
    # with glibc 2.31
    # 
    # Linking against an older version of glibc allows us
    # to produce a more portable binary
    nixpkgsGcc.url = "github:nixos/nixpkgs/nixos-unstable";

    # GPU drivers update frequently so we allow the version
    # of nixpkgs used for drivers to vary independently so
    # that we don't need to update everything else to get
    # the latest drivers
    nixpkgsDrv.url = "github:nixos/nixpkgs/nixos-unstable";

    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, nixpkgsDrv, nixpkgsGcc, rust-overlay, }:
    let
      system = "x86_64-linux";
      pkgsDrv = import nixpkgsDrv {
        inherit system;
        config.allowUnfree = true;
        config.cudaSupport = true;
      };
      pkgsGcc = import nixpkgsGcc {
        inherit system;
      };
      driverOverlay = final: prev: {
        cudaDrivers = pkgsDrv.linuxPackages.nvidia_x11;
      };
      gccOverlay = final: prev: {
        portableGcc = pkgsGcc.gcc;
      };
      overlays = [
        driverOverlay
        gccOverlay
        (import rust-overlay)
      ];
      pkgs = import nixpkgs {
        inherit system overlays;
        config.allowUnfree = true;
        config.cudaSupport = true;
      };
      shell = import ./nix/shell.nix { inherit pkgs; };
    in
    {
      formatter.${system} = pkgs.nixpkgs-fmt;
      devShells.${system}.default = shell;

    };
}
