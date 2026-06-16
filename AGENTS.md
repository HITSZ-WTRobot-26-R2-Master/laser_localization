# Repository Guidelines

## Project Structure & Module Organization

This repository contains the ROS 2 package `agv_pose_refiner`, built with `ament_cmake` and `ament_cmake_python`. Runtime code lives in `agv_pose_refiner_py/`; `agv_pose_refiner_node.py` is the main ROS 2 node that subscribes to coarse LiDAR poses and orchestrates the serial, solve, and publish layers. `common.py` holds shared dataclasses (`SensorMount`, `RangeFrame`, `CoarsePose`, `WallSegment`, `WallPair`, `BeamSelection`, `SolveResult`, etc.), protocol constants (`STP23L_*`), and utility functions (quaternion/euler conversion, 2D rotation/transform, CRC-16 Modbus, angle wrapping). `pose_solver.py` contains `PoseSolveLayer` — the core closed-form corner solver that matches coarse poses to map regions, selects beam triples from TOF sensors, computes refined pose in corner-local coordinates, and evaluates solution quality. `result_publisher.py` contains `ResultPublishLayer` — publishes refined pose to `/laser_pose`, status text to `/laser_status`, and optionally broadcasts `map` → `base_link` TF. `serial_receiver.py` contains `SerialReceiveLayer` — manages dual-threaded UART communication with STP23L TOF ranging sensors, decodes the binary protocol (0x5A 0xA5 header, CRC-16 Modbus), and maintains a fixed-size range frame buffer.

Configuration is split across three YAML files: `config/sensors.yaml` (mount positions and directions for 6 TOF sensors), `config/topics.yaml` (ROS topic names, serial port/baudrate/polling config, sensor-to-device mapping), `config/map_and_solver.yaml` (world geometry — walls, corners, wall pairs, scene profiles, solver tuning parameters).

ROS metadata is in `package.xml`, build/install rules are in `CMakeLists.txt`, Python packaging compatibility metadata is in `setup.py` and `setup.cfg`. Launch files live in `launch/`; `agv_pose_refiner.launch.py` starts the node with the three config file paths.

Do not add project logic to generated or workspace-level build output directories such as `build/`, `install/`, or `log/`. Keep node code in the Python package and configuration in `config/`. Do not define package-local pose/status messages here; use the shared `interfaces` package for `R2Pose`.

## Build, Test, and Development Commands

Use these ROS 2 commands only on a machine with a sourced ROS 2 environment, from the workspace root that contains this package:

```bash
colcon build --symlink-install --packages-select agv_pose_refiner
source install/setup.bash
ros2 launch agv_pose_refiner agv_pose_refiner.launch.py
```

This package depends on the shared `interfaces` package for `R2Pose`. It also requires `pyserial` for STP23L UART communication.

When dependencies are missing, install them through ROS package management from the workspace root:

```bash
rosdep install --from-paths . --ignore-src -r -y
```

## Coding Style & Naming Conventions

Use standard Python style with 4-space indentation, snake_case for functions, variables, and parameters, and PascalCase for classes. Keep ROS topic names and frame IDs explicit and stable. Prefer parameters from YAML config files over hard-coded values. Use `from __future__ import annotations` at the top of each module.

Sensor names follow the pattern `{position}_{direction}`: `front_center`, `rear_center`, `left_front`, `left_rear`, `right_front`, `right_rear`. Wall orientations use `vertical` (constant x, varies along y) or `horizontal` (constant y, varies along x). Yaw is expressed in degrees throughout the solver pipeline; conversions to radians happen at the math boundary.

## Testing Guidelines

Automated tests live under `test/`. Pure Python tests (no ROS imports) can run on any host:

```bash
python3 -m unittest discover -s test
```

On a machine with a sourced ROS 2 environment:

```bash
colcon test --packages-select agv_pose_refiner
colcon test-result --verbose
```

## Agent-Specific Instructions

This package is a ROS 2 Python node. The current machine does not have a ROS 2 environment, so do not run ROS build, ROS test, launch, dependency-install, or colcon compile-check commands here. Pure Python tests that do not require ROS imports may run on the host. Keep edits scoped to package code, configuration, and metadata.
