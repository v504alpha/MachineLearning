{
  inputs = {
    nixpkgs = {
      url = "github:nixos/nixpkgs/nixos-23.05";
    };
    flake-utils = {
      url = "github:numtide/flake-utils";
    };
  };
  outputs = { nixpkgs, flake-utils, ... }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs {
        inherit system;
      };
    in rec {
      devShell = pkgs.mkShell {
        buildInputs = with pkgs; [
            gcc
            gcc-unwrapped
          (python3.withPackages(ps: with ps; [
            ipython
            jupyter
            numpy
            pandas
            matplotlib
            tensorflow
            keras
            ipympl
          ]))
        ];
      };
    }
  );
}

