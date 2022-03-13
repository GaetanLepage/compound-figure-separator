{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {

    buildInputs = with pkgs; [
        python3

        stdenv.cc.cc

        # For Numpy
        zlib

        libGL

        glibc
    ];

    shellHook = ''
        # for PyTorch
        export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib

        # for Numpy
        export LD_LIBRARY_PATH=${pkgs.zlib}/lib:$LD_LIBRARY_PATH

        # GL libraries (for opencv)
        export LD_LIBRARY_PATH=${pkgs.libGL}/lib:$LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=${pkgs.glib.out}/lib:$LD_LIBRARY_PATH
    '';
}
