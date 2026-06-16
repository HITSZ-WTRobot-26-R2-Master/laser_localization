"""Tests for pose_solver.py that require ROS 2 (rclpy)."""

import unittest


class TestPoseSolverConfig(unittest.TestCase):
    """Basic structure tests — these validate the solver module is importable.

    Full solver tests need a ROS 2 environment (rclpy.init) and sensor hardware.
    They should be run via 'colcon test' with a sourced ROS 2 Humble overlay.
    """

    def test_import(self):
        from agv_pose_refiner_py import pose_solver

        self.assertTrue(hasattr(pose_solver, "PoseSolveLayer"))

    def test_state_constants(self):
        from agv_pose_refiner_py.pose_solver import (
            STATE_CANNOT_LOCALIZE,
            STATE_COARSE_ONLY,
            STATE_REFINED,
        )

        self.assertEqual(STATE_REFINED, "REFINED")
        self.assertEqual(STATE_COARSE_ONLY, "COARSE_ONLY")
        self.assertEqual(STATE_CANNOT_LOCALIZE, "CANNOT_LOCALIZE")


if __name__ == "__main__":
    unittest.main()
