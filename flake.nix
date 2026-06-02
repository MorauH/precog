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
        packages.default = pkgs.writeShellApplication {
          name = "precog";
          runtimeInputs = [ pkgs.uv pkgs.rosPackages.jazzy.python ];

          text = ''
            if [ -n "''${FORCE_CPU:-}" ]; then
              UV_EXTRA="cpu"
            elif command -v nvidia-smi >/dev/null 2>&1 || [ -c /dev/nvidia0 ]; then
              UV_EXTRA="cuda"
            else
              UV_EXTRA="cpu"
            fi

            exec uv run --extra "$UV_EXTRA" --python ${pkgs.rosPackages.jazzy.python}/bin/python3 precog "$@"
          '';
        };

        apps.default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/precog";
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [
            self.packages.${system}.default
            pkgs.rosPackages.jazzy.python
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

            # Dynamically detect hardware capability
            if [ -n "$FORCE_CPU" ]; then
              UV_EXTRA="cpu"
              echo "Force-CPU mode explicitly requested."
            elif command -v nvidia-smi &> /dev/null || [ -c /dev/nvidia0 ]; then
              UV_EXTRA="cuda"
              echo "NVIDIA Hardware detected. Provisioning CUDA environment..."
            else
              UV_EXTRA="cpu"
              echo "No NVIDIA GPU detected. Provisioning CPU-only environment..."
            fi

            precog() {
              uv run --extra "$UV_EXTRA" --python ${pkgs.rosPackages.jazzy.python}/bin/python3 precog "$@"
            }
            
            echo "Synchronizing virtual environment via uv..."
            uv sync --extra "$UV_EXTRA" --python ${pkgs.rosPackages.jazzy.python}/bin/python3
            
            source .venv/bin/activate

            echo "Environment ready. Verification:"
            python -c "import torch; print('  CUDA Available:', torch.cuda.is_available())"
            echo "   ros2 bag play recording.mcap --rate 2.0 --loop"
          '';
        };
      }
    );
}
