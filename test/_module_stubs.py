from __future__ import annotations

import sys
import types


def install_test_stubs() -> None:
    _install_ament_stubs()
    _install_rclpy_stubs()
    _install_geometry_msgs_stubs()
    _install_std_msgs_stubs()
    _install_tf2_ros_stubs()


def _install_ament_stubs() -> None:
    if "ament_index_python.packages" in sys.modules:
        return
    packages_module = types.ModuleType("ament_index_python.packages")

    class PackageNotFoundError(Exception):
        pass

    def get_package_share_directory(_package_name: str) -> str:
        return "."

    packages_module.PackageNotFoundError = PackageNotFoundError
    packages_module.get_package_share_directory = get_package_share_directory

    root_module = types.ModuleType("ament_index_python")
    root_module.packages = packages_module
    sys.modules["ament_index_python"] = root_module
    sys.modules["ament_index_python.packages"] = packages_module


def _install_rclpy_stubs() -> None:
    if "rclpy.time" in sys.modules:
        return
    time_module = types.ModuleType("rclpy.time")
    node_module = types.ModuleType("rclpy.node")

    class Time:
        def __init__(self, *, nanoseconds: int = 0) -> None:
            self.nanoseconds = nanoseconds

        @classmethod
        def from_msg(cls, _msg: object) -> "Time":
            return cls()

        def to_msg(self) -> None:
            return None

    class Node:
        pass

    time_module.Time = Time
    node_module.Node = Node
    root_module = types.ModuleType("rclpy")
    root_module.time = time_module
    root_module.node = node_module
    root_module.ok = lambda: True
    root_module.shutdown = lambda: None
    sys.modules["rclpy"] = root_module
    sys.modules["rclpy.time"] = time_module
    sys.modules["rclpy.node"] = node_module


def _install_geometry_msgs_stubs() -> None:
    if "geometry_msgs.msg" in sys.modules:
        return
    geometry_msgs_msg_module = types.ModuleType("geometry_msgs.msg")

    class _Header:
        def __init__(self) -> None:
            self.frame_id = ""
            self.stamp = None

    class _Vector3:
        def __init__(self) -> None:
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    class _Quaternion:
        def __init__(self) -> None:
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
            self.w = 1.0

    class _Transform:
        def __init__(self) -> None:
            self.translation = _Vector3()
            self.rotation = _Quaternion()

    class TransformStamped:
        def __init__(self) -> None:
            self.header = _Header()
            self.child_frame_id = ""
            self.transform = _Transform()

    geometry_msgs_msg_module.TransformStamped = TransformStamped

    geometry_msgs_module = types.ModuleType("geometry_msgs")
    geometry_msgs_module.msg = geometry_msgs_msg_module
    sys.modules["geometry_msgs"] = geometry_msgs_module
    sys.modules["geometry_msgs.msg"] = geometry_msgs_msg_module


def _install_std_msgs_stubs() -> None:
    if "std_msgs.msg" in sys.modules:
        return
    std_msgs_msg_module = types.ModuleType("std_msgs.msg")

    class String:
        def __init__(self) -> None:
            self.data = ""

    class UInt8:
        def __init__(self) -> None:
            self.data = 0

    std_msgs_msg_module.String = String
    std_msgs_msg_module.UInt8 = UInt8

    std_msgs_module = types.ModuleType("std_msgs")
    std_msgs_module.msg = std_msgs_msg_module
    sys.modules["std_msgs"] = std_msgs_module
    sys.modules["std_msgs.msg"] = std_msgs_msg_module


def _install_tf2_ros_stubs() -> None:
    if "tf2_ros" in sys.modules:
        return
    tf2_ros_module = types.ModuleType("tf2_ros")

    class TransformBroadcaster:
        def __init__(self, _node: object) -> None:
            self.transforms = []

        def sendTransform(self, transform: object) -> None:
            self.transforms.append(transform)

    tf2_ros_module.TransformBroadcaster = TransformBroadcaster
    sys.modules["tf2_ros"] = tf2_ros_module
