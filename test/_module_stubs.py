from __future__ import annotations

import sys
import types


def install_test_stubs() -> None:
    _install_ament_stubs()
    _install_rclpy_stubs()


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

    class Time:
        def __init__(self, *, nanoseconds: int = 0) -> None:
            self.nanoseconds = nanoseconds

        @classmethod
        def from_msg(cls, _msg: object) -> "Time":
            return cls()

        def to_msg(self) -> None:
            return None

    time_module.Time = Time
    root_module = types.ModuleType("rclpy")
    root_module.time = time_module
    sys.modules["rclpy"] = root_module
    sys.modules["rclpy.time"] = time_module
