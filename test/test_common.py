"""Tests for common.py utility functions that do NOT require ROS imports."""

import math
import unittest

from agv_pose_refiner_py.common import (
    coerce_bool,
    crc16_modbus,
    euler_from_quaternion_components,
    quaternion_components_from_rpy,
    rotate_2d,
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
        self.assertAlmostEqual(yaw, 0.0)

    def test_yaw_accumulation(self):
        x, y, yaw = transform_pose_2d(0.0, 0.0, 45.0, 0.0, 0.0, 30.0)
        self.assertAlmostEqual(yaw, 75.0)


class TestCrc16Modbus(unittest.TestCase):
    def test_known_vector(self):
        data = bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x02])
        crc = crc16_modbus(data)
        self.assertEqual(crc, 0xC40B)

    def test_empty(self):
        self.assertEqual(crc16_modbus(b""), 0xFFFF)


if __name__ == "__main__":
    unittest.main()
