from __future__ import annotations

import math
import unittest

from _module_stubs import install_test_stubs

install_test_stubs()

from rclpy.time import Time

from agv_pose_refiner_py.common import (
    INFRARED_QUERY_COMMAND,
    RangeFrame,
    SENSOR_ORDER,
    SensorMount,
    SerialSensorMapping,
    STP23L_DATA_COMMAND,
    STP23L_QUERY_COMMAND,
    crc16_modbus,
)
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


class DummyInfraredLayer:
    def __init__(self) -> None:
        self.reset_count = 0
        self.frames = []
        self.query_calls = 0

    def reset_shared_serial_state(self) -> None:
        self.reset_count += 1

    def maybe_send_queries(self, _handle: object) -> None:
        self.query_calls += 1

    def handle_infrared_frame(self, frame: object) -> None:
        self.frames.append(frame)


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
        self.infrared_layer = DummyInfraredLayer()
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
            infrared_layer=self.infrared_layer,
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
        self.assertEqual(self.infrared_layer.reset_count, 1)

    def test_shared_buffer_parses_infrared_and_laser_frames(self) -> None:
        infrared_frame = self._build_infrared_data_frame(
            device_id=3,
            report_type=0x01,
            raw_byte=0x11,
            device_timestamp_ms=123,
        )
        laser_frame = self._build_laser_data_frame(
            device_id=1,
            report_type=0x01,
            status_bits=0x07,
            distances_mm=[100, 200, 300, 0xFFFF],
        )

        with self.layer._parser_lock:
            self.layer._serial_byte_buffer.extend(infrared_frame + laser_frame)
            produced = self.layer._drain_serial_buffer_locked()

        self.assertTrue(produced)
        self.assertEqual(len(self.infrared_layer.frames), 1)
        self.assertEqual(self.infrared_layer.frames[0].device_id, 3)
        self.assertEqual(self.infrared_layer.frames[0].raw_byte, 0x11)
        self.assertIn(1, self.layer._pending_board_frames)
        self.assertEqual(self.layer._pending_board_frames[1].distances_mm[:3], [100, 200, 300])
        self.assertEqual(self.layer._serial_byte_buffer, bytearray())

    def test_shared_buffer_discards_infrared_query_echo(self) -> None:
        query_echo = self._build_query_frame(INFRARED_QUERY_COMMAND, device_id=3)
        laser_query_echo = self._build_query_frame(STP23L_QUERY_COMMAND, device_id=1)

        with self.layer._parser_lock:
            self.layer._serial_byte_buffer.extend(query_echo + laser_query_echo)
            produced = self.layer._drain_serial_buffer_locked()

        self.assertFalse(produced)
        self.assertEqual(self.layer._serial_byte_buffer, bytearray())

    def _build_query_frame(self, command: int, *, device_id: int) -> bytes:
        payload = bytes([0x5A, 0xA5, command, device_id & 0xFF])
        crc = crc16_modbus(payload)
        return payload + crc.to_bytes(2, byteorder="little")

    def _build_infrared_data_frame(
        self,
        *,
        device_id: int,
        report_type: int,
        raw_byte: int,
        device_timestamp_ms: int,
    ) -> bytes:
        payload = bytes(
            [
                0x5A,
                0xA5,
                0x82,
                device_id & 0xFF,
                report_type & 0xFF,
                raw_byte & 0xFF,
            ]
        ) + int(device_timestamp_ms).to_bytes(4, byteorder="little")
        crc = crc16_modbus(payload)
        return payload + crc.to_bytes(2, byteorder="little")

    def _build_laser_data_frame(
        self,
        *,
        device_id: int,
        report_type: int,
        status_bits: int,
        distances_mm: list[int],
    ) -> bytes:
        payload = bytes(
            [
                0x5A,
                0xA5,
                STP23L_DATA_COMMAND,
                device_id & 0xFF,
                report_type & 0xFF,
                status_bits & 0xFF,
            ]
        ) + b"".join(
            int(distance_mm).to_bytes(2, byteorder="little")
            for distance_mm in distances_mm
        )
        crc = crc16_modbus(payload)
        return payload + crc.to_bytes(2, byteorder="little")


if __name__ == "__main__":
    unittest.main()
