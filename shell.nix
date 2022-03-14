{ pkgs ? import <nixpkgs> {} }:


pkgs.mkShell {

    buildInputs = with pkgs; [

        python3

        stdenv.cc.cc

        # For Numpy
        zlib

        libGL

        xlibs.libSM
        xlibs.libICE
        xlibs.libXcursor

        qt5.full
        qt5.qtbase
    ];

    shellHook = ''
        # for PyTorch
        export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib

        # for Numpy
        export LD_LIBRARY_PATH=${pkgs.zlib}/lib:$LD_LIBRARY_PATH

        # GL libraries (for opencv)
        export LD_LIBRARY_PATH=${pkgs.libGL}/lib:$LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=${pkgs.glib.out}/lib:$LD_LIBRARY_PATH

        export LD_LIBRARY_PATH=${pkgs.xlibs.libSM.out}/lib:$LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=${pkgs.xlibs.libICE.out}/lib:$LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=${pkgs.xlibs.libXcursor.out}/lib:$LD_LIBRARY_PATH
        export QT_QPA_PLATFORM_PLUGIN_PATH=${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins
    '';
}
