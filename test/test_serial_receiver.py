from __future__ import annotations

import math
import unittest

from _module_stubs import install_test_stubs

install_test_stubs()

from rclpy.time import Time

from agv_pose_refiner_py.common import RangeFrame, SensorMount, SerialSensorMapping
from agv_pose_refiner_py.serial_receiver import SerialReceiveLayer


class DummyLogger:
    def info(self, _msg: str) -> None:
        pass

    def warn(self, _msg: str) -> None:
        pass

    def error(self, _msg: str) -> None:
        pass


class DummyClock:
    def __init__(self) -> None:
        self.clock_type = 0
        self.now_time = Time(nanoseconds=0)

    def now(self) -> Time:
        return self.now_time


class DummyNode:
    def __init__(self) -> None:
        self._logger = DummyLogger()
        self._clock = DummyClock()

    def get_logger(self) -> DummyLogger:
        return self._logger

    def get_clock(self) -> DummyClock:
        return self._clock


class DummyHandle:
    def reset_input_buffer(self) -> None:
        return None

    def reset_output_buffer(self) -> None:
        return None


def _build_sensor_mount() -> SensorMount:
    return SensorMount(
        pos_x=0.0,
        pos_y=0.0,
        dir_x=1.0,
        dir_y=0.0,
        min_range_m=0.03,
        max_range_m=2.0,
    )


class TestSerialReceiveLayer(unittest.TestCase):
    def setUp(self) -> None:
        sensor_mounts = {
            "front_center": _build_sensor_mount(),
            "rear_center": _build_sensor_mount(),
            "left_front": _build_sensor_mount(),
            "left_rear": _build_sensor_mount(),
            "right_front": _build_sensor_mount(),
            "right_rear": _build_sensor_mount(),
        }
        sensor_map = {
            "front_center": SerialSensorMapping("front_center", 1, 0),
            "rear_center": SerialSensorMapping("rear_center", 2, 0),
            "left_front": SerialSensorMapping("left_front", 1, 1),
            "left_rear": SerialSensorMapping("left_rear", 1, 2),
            "right_front": SerialSensorMapping("right_front", 2, 1),
            "right_rear": SerialSensorMapping("right_rear", 2, 2),
        }
        self.node = DummyNode()
        self.layer = SerialReceiveLayer(
            node=self.node,
            sensor_mounts=sensor_mounts,
            sensor_map=sensor_map,
            query_device_ids=[1, 2],
            serial_port="/dev/laser_serial",
            serial_baudrate=115200,
            serial_timeout_sec=0.02,
            serial_min_publish_interval_ms=5.0,
            serial_poll_rate_hz=10.0,
            serial_response_timeout_sec=0.02,
            serial_decode_log_enabled=True,
            serial_decode_log_interval_ms=0.0,
            serial_expect_matching_device_id=True,
        )

    def test_snapshot_frame_drops_stale_frame(self) -> None:
        frame = RangeFrame(
            stamp=Time(nanoseconds=1_000_000_000),
            ranges={name: 0.5 for name in self.layer._sensor_map.keys()},
            valid={name: True for name in self.layer._sensor_map.keys()},
        )
        self.layer._latest_range_frame = frame
        self.node.get_clock().now_time = Time(nanoseconds=1_500_000_000)

        self.assertIs(
            self.layer.snapshot_frame(now=self.node.get_clock().now(), max_age_ms=600.0),
            frame,
        )
        self.assertIsNone(
            self.layer.snapshot_frame(now=self.node.get_clock().now(), max_age_ms=400.0)
        )

    def test_prepare_serial_handle_clears_latest_published_state(self) -> None:
        self.layer._latest_range_frame = RangeFrame(
            stamp=Time(nanoseconds=1_000_000_000),
            ranges={name: 0.5 for name in self.layer._sensor_map.keys()},
            valid={name: True for name in self.layer._sensor_map.keys()},
        )
        self.layer._last_serial_frame_stamp = Time(nanoseconds=1_000_000_000)
        self.layer._latest_serial_ranges_m["front_center"] = 0.5
        self.layer._latest_serial_valid["front_center"] = True

        self.layer._prepare_serial_handle(DummyHandle())

        self.assertIsNone(self.layer.snapshot_frame())
        self.assertIsNone(self.layer._last_serial_frame_stamp)
        self.assertTrue(math.isnan(self.layer._latest_serial_ranges_m["front_center"]))
        self.assertFalse(self.layer._latest_serial_valid["front_center"])


if __name__ == "__main__":
    unittest.main()
