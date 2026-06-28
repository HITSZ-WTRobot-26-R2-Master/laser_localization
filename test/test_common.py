"""Tests for common.py utility functions that do NOT require ROS imports."""

import math
import unittest

from _module_stubs import install_test_stubs

install_test_stubs()

from agv_pose_refiner_py.common import (
    coerce_bool,
    crc16_modbus,
    euler_from_quaternion_components,
    quaternion_components_from_rpy,
    resolve_query_device_ids,
    resolve_serial_max_range_frame_age_ms,
    rotate_2d,
    SerialSensorMapping,
    transform_pose_2d,
    wrap_deg,
)


class TestWrapDeg(unittest.TestCase):
    def test_no_wrap(self):
        self.assertAlmostEqual(wrap_deg(0.0), 0.0)
        self.assertAlmostEqual(wrap_deg(45.0), 45.0)
        self.assertAlmostEqual(wrap_deg(-45.0), -45.0)
        self.assertAlmostEqual(wrap_deg(179.0), 179.0)
        self.assertAlmostEqual(wrap_deg(-179.0), -179.0)

    def test_positive_wrap(self):
        self.assertAlmostEqual(wrap_deg(181.0), -179.0)
        self.assertAlmostEqual(wrap_deg(360.0), 0.0)
        self.assertAlmostEqual(wrap_deg(540.0), 180.0)
        self.assertAlmostEqual(wrap_deg(720.0), 0.0)

    def test_negative_wrap(self):
        self.assertAlmostEqual(wrap_deg(-181.0), 179.0)
        self.assertAlmostEqual(wrap_deg(-360.0), 0.0)
        self.assertAlmostEqual(wrap_deg(-540.0), -180.0)


class TestCoerceBool(unittest.TestCase):
    def test_bool_input(self):
        self.assertTrue(coerce_bool(True))
        self.assertFalse(coerce_bool(False))

    def test_string_input(self):
        self.assertTrue(coerce_bool("true"))
        self.assertTrue(coerce_bool("True"))
        self.assertTrue(coerce_bool("1"))
        self.assertTrue(coerce_bool("yes"))
        self.assertTrue(coerce_bool("on"))
        self.assertFalse(coerce_bool("false"))
        self.assertFalse(coerce_bool("0"))
        self.assertFalse(coerce_bool("no"))
        self.assertFalse(coerce_bool(""))

    def test_numeric_input(self):
        self.assertTrue(coerce_bool(1))
        self.assertFalse(coerce_bool(0))


class TestEulerQuaternionRoundTrip(unittest.TestCase):
    def test_identity(self):
        qx, qy, qz, qw = quaternion_components_from_rpy(0.0, 0.0, 0.0)
        roll, pitch, yaw = euler_from_quaternion_components(qx, qy, qz, qw)
        self.assertAlmostEqual(roll, 0.0)
        self.assertAlmostEqual(pitch, 0.0)
        self.assertAlmostEqual(yaw, 0.0)

    def test_yaw_90(self):
        qx, qy, qz, qw = quaternion_components_from_rpy(0.0, 0.0, math.pi / 2.0)
        roll, pitch, yaw = euler_from_quaternion_components(qx, qy, qz, qw)
        self.assertAlmostEqual(roll, 0.0)
        self.assertAlmostEqual(pitch, 0.0)
        self.assertAlmostEqual(yaw, math.pi / 2.0)

    def test_yaw_neg_90(self):
        qx, qy, qz, qw = quaternion_components_from_rpy(0.0, 0.0, -math.pi / 2.0)
        roll, pitch, yaw = euler_from_quaternion_components(qx, qy, qz, qw)
        self.assertAlmostEqual(roll, 0.0)
        self.assertAlmostEqual(pitch, 0.0)
        self.assertAlmostEqual(yaw, -math.pi / 2.0)

    def test_roll_45(self):
        qx, qy, qz, qw = quaternion_components_from_rpy(math.pi / 4.0, 0.0, 0.0)
        roll, pitch, yaw = euler_from_quaternion_components(qx, qy, qz, qw)
        self.assertAlmostEqual(roll, math.pi / 4.0)
        self.assertAlmostEqual(pitch, 0.0)
        self.assertAlmostEqual(yaw, 0.0)

    def test_pitch_30(self):
        qx, qy, qz, qw = quaternion_components_from_rpy(0.0, math.pi / 6.0, 0.0)
        roll, pitch, yaw = euler_from_quaternion_components(qx, qy, qz, qw)
        self.assertAlmostEqual(roll, 0.0)
        self.assertAlmostEqual(pitch, math.pi / 6.0)
        self.assertAlmostEqual(yaw, 0.0)


class TestRotate2D(unittest.TestCase):
    def test_identity(self):
        x, y = rotate_2d(1.0, 2.0, 0.0)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 2.0)

    def test_90_deg(self):
        x, y = rotate_2d(1.0, 0.0, 90.0)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 1.0)

    def test_neg_90_deg(self):
        x, y = rotate_2d(1.0, 0.0, -90.0)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, -1.0)

    def test_180_deg(self):
        x, y = rotate_2d(1.0, 2.0, 180.0)
        self.assertAlmostEqual(x, -1.0)
        self.assertAlmostEqual(y, -2.0)


class TestTransformPose2D(unittest.TestCase):
    def test_origin_translation(self):
        x, y, yaw = transform_pose_2d(0.0, 0.0, 0.0, 5.0, 0.0, 0.0)
        self.assertAlmostEqual(x, 5.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(yaw, 0.0)

    def test_rotated_frame_translation(self):
        x, y, yaw = transform_pose_2d(0.0, 0.0, 90.0, 1.0, 0.0, 0.0)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 1.0)
        self.assertAlmostEqual(yaw, 90.0)

    def test_yaw_accumulation(self):
        x, y, yaw = transform_pose_2d(0.0, 0.0, 45.0, 0.0, 0.0, 30.0)
        self.assertAlmostEqual(yaw, 75.0)


class TestCrc16Modbus(unittest.TestCase):
    def test_known_vector(self):
        data = bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x02])
        crc = crc16_modbus(data)
        self.assertEqual(crc, 0x0BC4)

    def test_empty(self):
        self.assertEqual(crc16_modbus(b""), 0xFFFF)


class TestResolveQueryDeviceIds(unittest.TestCase):
    def setUp(self):
        self.sensor_map = {
            "front_center": SerialSensorMapping("front_center", 3, 0),
            "rear_center": SerialSensorMapping("rear_center", 4, 0),
            "left_front": SerialSensorMapping("left_front", 3, 1),
            "left_rear": SerialSensorMapping("left_rear", 3, 2),
            "right_front": SerialSensorMapping("right_front", 4, 1),
            "right_rear": SerialSensorMapping("right_rear", 4, 2),
        }

    def test_defaults_to_sensor_map_device_ids(self):
        self.assertEqual(resolve_query_device_ids([], self.sensor_map), [3, 4])
        self.assertEqual(resolve_query_device_ids(None, self.sensor_map), [3, 4])

    def test_rejects_override_mismatching_sensor_map(self):
        with self.assertRaisesRegex(RuntimeError, "must match the device ids used in sensor_map"):
            resolve_query_device_ids([1, 2], self.sensor_map)


class TestResolveSerialMaxRangeFrameAgeMs(unittest.TestCase):
    def test_positive_value_is_used_directly(self):
        self.assertAlmostEqual(
            resolve_serial_max_range_frame_age_ms(120.0, 10.0),
            120.0,
        )

    def test_zero_value_is_derived_from_poll_rate(self):
        self.assertAlmostEqual(
            resolve_serial_max_range_frame_age_ms(0.0, 10.0),
            250.0,
        )
        self.assertAlmostEqual(
            resolve_serial_max_range_frame_age_ms(0.0, 2.0),
            1250.0,
        )


if __name__ == "__main__":
    unittest.main()
