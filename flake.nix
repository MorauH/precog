{
  description = "Python ML development environment with uv, PyTorch and MCAP Replayer";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nix-ros-overlay.url = "github:lopsided98/nix-ros-overlay";
    nix-ros-overlay.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, nix-ros-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ 
            nix-ros-overlay.overlays.default
            (self: super: {
              vcstool = super.vcs2l;   # temporary alias
              rosPackages = super.rosPackages // {
                jazzy = super.rosPackages.jazzy // {
                  cfs-msgs = super.rosPackages.jazzy.callPackage ./cfs-msgs.nix { };
                };
              };
            })
          ];
        };

        ros = pkgs.rosPackages.jazzy;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.python312
            pkgs.uv

            # ROS 2 Core
            ros.ros2cli
            ros.ros2cli-common-extensions

            ros.rosbag2
            ros.rosbag2-storage-mcap
            ros.rclpy
            
            # Essential message packages
            ros.std-msgs
            ros.std-srvs
            ros.sensor-msgs
            ros.geometry-msgs
            ros.nav-msgs
            ros.visualization-msgs
            ros.message-filters

            ros.cfs-msgs

            # Expected by ROS2
            pkgs.python3Packages.pyyaml
            pkgs.python3Packages.setuptools
          ];

          LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ]}:/usr/lib/wsl/lib";

          shellHook = ''
            export PATH="/usr/lib/wsl/lib:$PATH"
            echo "Python ML + MCAP Replayer environment loaded"
            echo "Usage:"
            echo "   ros2 bag play recording.mcap --rate 2.0 --loop"
            echo "   ros2 bag play recording.mcap --topics /imu /odom"
            echo ""            
            
            if [ ! -d .venv ]; then
              echo "Creating virtual environment..."
              uv venv .venv --python python3
            fi

            source .venv/bin/activate

            if ! python -c "import mcap" 2>/dev/null; then
              uv pip install mcap mcap-ros2-support
            fi

            echo "Ready: torch, mcap-replayer, etc."
          '';
        };
      }
    );
}
