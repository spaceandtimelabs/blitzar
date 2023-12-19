{
  description = "blitzar build environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    nixpkgsGcc.url = "github:nixos/nixpkgs/nixos-unstable";
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

      # WIP: Set up a docker image
      #
      # To build the image, run
      #   > nix build .#docker
      #   > sudo docker load <result
      # The image can be run with
      #   > docker 
      #   > sudo docker run -v /home/rnburn/proj/blitzar:/src -w /src --rm --runtime=nvidia --gpus all -it blitzar:<hash>
      # TODO(rnburn):
      #   * Invoking 
      #        bazel build //sxt/...
      #     in the container should work but the full build
      #        bazel build //...
      #     needs a bit of work to get the benchmarks to build
      #   * When running
      #        bazel test //sxt/...
      #     the GPU tests will fail -- some steps are missing in the docker environment to get the
      #     GPU drivers to appear
      packages.${system}.docker = 
        let
          clang = import ./nix/clang.nix { inherit pkgs; };
          cuda = import ./nix/cuda.nix { inherit pkgs; };
          bazel = import ./nix/bazel.nix { inherit pkgs; inherit clang; inherit cuda; };
        in
        with pkgs;
        pkgs.dockerTools.buildImage {
          name = "blitzar";
          copyToRoot = with pkgs; [
            bazel
            binutils
            coreutils
            dockerTools.usrBinEnv
            dockerTools.binSh
            dockerTools.caCertificates
            dockerTools.fakeNss
            pkgs.nvidia-docker
          ];

          config = {
            Cmd = [
              "/bin/sh"
            ];
          };
        };
    };
}
