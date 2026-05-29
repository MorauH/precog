{ buildRosPackage
, ament-cmake
, rosidl-default-generators
, rosidl-default-runtime
, geometry-msgs
, std-msgs
, sensor-msgs
, action-msgs
, builtin-interfaces
, ...
}:

buildRosPackage {
  pname = "cfs-msgs";
  version = "0.0.0";

  src = builtins.fetchGit {
    url = "git@gitlab.com:chalmersfs/software/computer/ros-packages/cfs_msgs.git";
    rev = "f15455229916e247c62bbefc9da1f6461b25ca31";
  };

  buildInputs = [
    ament-cmake
    rosidl-default-generators
  ];

  propagatedBuildInputs = [
    rosidl-default-runtime
    geometry-msgs
    std-msgs
    sensor-msgs
    action-msgs
    builtin-interfaces
  ];

  meta = {
    description = "Custom messages for CFS";
  };
}
