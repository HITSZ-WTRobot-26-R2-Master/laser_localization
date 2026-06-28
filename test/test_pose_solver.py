from __future__ import annotations

import math
import unittest

from _module_stubs import install_test_stubs

install_test_stubs()

from rclpy.time import Time

from agv_pose_refiner_py.common import (
    SENSOR_ORDER,
    CoarsePose,
    RangeFrame,
    SensorGeometry,
    SensorMount,
    SolveResult,
    WallPair,
    WallSegment,
)
from agv_pose_refiner_py.pose_solver import (
    STATE_CANNOT_LOCALIZE,
    STATE_COARSE_ONLY,
    STATE_REFINED,
    PoseSolveLayer,
)


def _build_result(
    *,
    state: str,
    reason: str,
    wall_pair_name: str,
    residual_m: float | None,
    target_hit_count: int,
    selected_valid_beam_count: int,
    delta_xy_norm_m: float | None,
    delta_yaw_deg: float | None,
    score: float,
) -> SolveResult:
    correction_debug = {}
    if delta_xy_norm_m is not None:
        correction_debug["delta_xy_norm_m"] = delta_xy_norm_m
    if delta_yaw_deg is not None:
        correction_debug["delta_yaw_deg"] = delta_yaw_deg
    return SolveResult(
        state=state,
        reason=reason,
        pose_source="test",
        localized=state == STATE_REFINED,
        x=0.0,
        y=0.0,
        yaw_deg=0.0,
        valid_beam_count=selected_valid_beam_count,
        score=score,
        prior_age_ms=0.0,
        selected_valid_beam_count=selected_valid_beam_count,
        target_hit_count=target_hit_count,
        debug={"solver_debug": {"correction_debug": correction_debug}},
        residual_m=residual_m,
        wall_pair_name=wall_pair_name,
    )


class TestPoseSolverConfig(unittest.TestCase):
    def test_import(self) -> None:
        self.assertTrue(hasattr(PoseSolveLayer, "_select_region_match_with_debug"))

    def test_state_constants(self) -> None:
        self.assertEqual(STATE_REFINED, "REFINED")
        self.assertEqual(STATE_COARSE_ONLY, "COARSE_ONLY")
        self.assertEqual(STATE_CANNOT_LOCALIZE, "CANNOT_LOCALIZE")


class TestDualWallPairRegionSupport(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = PoseSolveLayer.__new__(PoseSolveLayer)
        self.solver.wall_pairs = {
            "pair_a": WallPair(
                name="pair_a",
                x_wall_name="x_wall",
                x_wall_role="front",
                side_wall_name="side_wall",
                side_wall_role="right",
                corner_x=0.0,
                corner_y=0.0,
                corner_yaw_deg=0.0,
            ),
            "pair_b": WallPair(
                name="pair_b",
                x_wall_name="x_wall",
                x_wall_role="rear",
                side_wall_name="side_wall",
                side_wall_role="left",
                corner_x=0.0,
                corner_y=0.0,
                corner_yaw_deg=180.0,
            ),
        }

    def test_validate_scene_profiles_accepts_dual_wall_pair_region(self) -> None:
        scene_profiles = {
            "scene": {
                "wall_selector": {
                    "regions": [
                        {
                            "name": "special_region",
                            "x_range": [1.0, 2.0],
                            "y_range": [3.0, 4.0],
                            "active_wall_pairs": ["pair_a", "pair_b"],
                        }
                    ]
                }
            }
        }

        self.solver._validate_scene_profiles(scene_profiles)

    def test_validate_scene_profiles_accepts_special_solver_region(self) -> None:
        scene_profiles = {
            "scene": {
                "wall_selector": {
                    "regions": [
                        {
                            "name": "special_region",
                            "x_range": [1.0, 2.0],
                            "y_range": [3.0, 4.0],
                            "active_wall_pair": "pair_a",
                            "special_solver": {
                                "type": "compensated_front_side",
                                "compensation_x0_m": 0.125,
                                "max_iterations": 8,
                                "theta_tolerance_deg": 0.01,
                            },
                        }
                    ]
                }
            }
        }

        self.solver._validate_scene_profiles(scene_profiles)

    def test_validate_scene_profiles_accepts_projected_xy_special_solver_region(
        self,
    ) -> None:
        scene_profiles = {
            "scene": {
                "wall_selector": {
                    "regions": [
                        {
                            "name": "special_region",
                            "x_range": [1.0, 2.0],
                            "y_range": [3.0, 4.0],
                            "active_wall_pair": "pair_a",
                            "special_solver": {
                                "type": "projected_xy_with_lidar_yaw",
                                "solve_axes": ["x", "y"],
                                "x_beam": "right_front",
                                "y_beam": "rear_center",
                                "min_dir_component_abs": 0.2,
                                "max_x_correction_m": 0.2,
                                "max_y_correction_m": 0.2,
                            },
                        }
                    ]
                }
            }
        }
        self.solver.walls = {
            "x_wall": WallSegment(
                name="x_wall",
                orientation="horizontal",
                const_value=1.2,
                min_axis=0.0,
                max_axis=3.2,
            ),
            "side_wall": WallSegment(
                name="side_wall",
                orientation="vertical",
                const_value=3.2,
                min_axis=-2.0,
                max_axis=1.2,
            ),
        }

        self.solver._validate_scene_profiles(scene_profiles)

    def test_validate_scene_profiles_rejects_invalid_xy_yaw_source(self) -> None:
        scene_profiles = {
            "scene": {
                "wall_selector": {
                    "regions": [
                        {
                            "name": "special_region",
                            "x_range": [1.0, 2.0],
                            "y_range": [3.0, 4.0],
                            "active_wall_pair": "pair_a",
                            "xy_yaw_source": "imu",
                        }
                    ]
                }
            }
        }

        with self.assertRaisesRegex(
            RuntimeError, "xy_yaw_source must be side_laser or lidar"
        ):
            self.solver._validate_scene_profiles(scene_profiles)

    def test_select_region_match_returns_both_wall_pairs(self) -> None:
        coarse = CoarsePose(
            stamp=Time(nanoseconds=0),
            x=1.5,
            y=3.5,
            z=0.0,
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_deg=0.0,
        )
        scene_cfg = {
            "wall_selector": {
                "regions": [
                    {
                        "name": "special_region",
                        "x_range": [1.0, 2.0],
                        "y_range": [3.0, 4.0],
                        "active_wall_pairs": ["pair_a", "pair_b"],
                        "priority": 100,
                    }
                ]
            }
        }

        region_match, region_debug = self.solver._select_region_match_with_debug(
            scene_cfg, coarse
        )

        self.assertIsNotNone(region_match)
        self.assertEqual(region_match.name, "special_region")
        self.assertEqual(
            [wall_pair.name for wall_pair in region_match.wall_pairs],
            ["pair_a", "pair_b"],
        )
        self.assertEqual(
            region_debug["matched_wall_pair_names"],
            ["pair_a", "pair_b"],
        )

    def test_select_region_match_preserves_special_solver_config(self) -> None:
        coarse = CoarsePose(
            stamp=Time(nanoseconds=0),
            x=1.5,
            y=3.5,
            z=0.0,
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_deg=0.0,
        )
        scene_cfg = {
            "wall_selector": {
                "regions": [
                    {
                        "name": "special_region",
                        "x_range": [1.0, 2.0],
                        "y_range": [3.0, 4.0],
                        "active_wall_pair": "pair_a",
                        "special_solver": {
                            "type": "compensated_front_side",
                            "compensation_x0_m": 0.125,
                        },
                    }
                ]
            }
        }

        region_match, region_debug = self.solver._select_region_match_with_debug(
            scene_cfg, coarse
        )

        self.assertIsNotNone(region_match)
        self.assertEqual(region_match.name, "special_region")
        self.assertIsNotNone(region_match.region_config)
        self.assertEqual(
            region_match.region_config["special_solver"]["type"],
            "compensated_front_side",
        )
        self.assertEqual(
            region_debug["matched_special_solver_type"],
            "compensated_front_side",
        )

    def test_select_best_solver_candidate_prefers_refined_result(self) -> None:
        candidate_results = [
            _build_result(
                state=STATE_COARSE_ONLY,
                reason="INSUFFICIENT_VALID_BEAMS",
                wall_pair_name="pair_a",
                residual_m=None,
                target_hit_count=0,
                selected_valid_beam_count=2,
                delta_xy_norm_m=None,
                delta_yaw_deg=None,
                score=0.0,
            ),
            _build_result(
                state=STATE_REFINED,
                reason="OK",
                wall_pair_name="pair_b",
                residual_m=0.018,
                target_hit_count=3,
                selected_valid_beam_count=3,
                delta_xy_norm_m=0.03,
                delta_yaw_deg=1.2,
                score=0.4,
            ),
        ]

        selected_index = self.solver._select_best_solver_candidate_index(
            candidate_results
        )

        self.assertEqual(selected_index, 1)

    def test_select_best_solver_candidate_uses_lidar_distance_as_tie_breaker(self) -> None:
        candidate_results = [
            _build_result(
                state=STATE_REFINED,
                reason="OK",
                wall_pair_name="pair_a",
                residual_m=0.01,
                target_hit_count=3,
                selected_valid_beam_count=3,
                delta_xy_norm_m=0.08,
                delta_yaw_deg=3.0,
                score=0.67,
            ),
            _build_result(
                state=STATE_REFINED,
                reason="OK",
                wall_pair_name="pair_b",
                residual_m=0.01,
                target_hit_count=3,
                selected_valid_beam_count=3,
                delta_xy_norm_m=0.02,
                delta_yaw_deg=0.8,
                score=0.67,
            ),
        ]

        selected_index = self.solver._select_best_solver_candidate_index(
            candidate_results
        )

        self.assertEqual(selected_index, 1)


class TestRegionYawSourceSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = PoseSolveLayer.__new__(PoseSolveLayer)
        pair_spacing_m = 0.45816
        half_pair_spacing_m = pair_spacing_m * 0.5
        zero_mount = dict(min_range_m=0.01, max_range_m=20.0)
        self.solver.sensor_mounts = {
            "front_center": SensorMount(
                pos_x=0.0,
                pos_y=0.0,
                dir_x=1.0,
                dir_y=0.0,
                **zero_mount,
            ),
            "rear_center": SensorMount(
                pos_x=0.0,
                pos_y=0.0,
                dir_x=-1.0,
                dir_y=0.0,
                **zero_mount,
            ),
            "left_front": SensorMount(
                pos_x=half_pair_spacing_m,
                pos_y=0.0,
                dir_x=0.0,
                dir_y=1.0,
                **zero_mount,
            ),
            "left_rear": SensorMount(
                pos_x=-half_pair_spacing_m,
                pos_y=0.0,
                dir_x=0.0,
                dir_y=1.0,
                **zero_mount,
            ),
            "right_front": SensorMount(
                pos_x=half_pair_spacing_m,
                pos_y=0.0,
                dir_x=0.0,
                dir_y=-1.0,
                **zero_mount,
            ),
            "right_rear": SensorMount(
                pos_x=-half_pair_spacing_m,
                pos_y=0.0,
                dir_x=0.0,
                dir_y=-1.0,
                **zero_mount,
            ),
        }
        self.solver.sensor_geometry = SensorGeometry(
            x_front=0.0,
            x_rear=0.0,
            y_left=0.0,
            y_right=0.0,
            x_left_pair=pair_spacing_m,
            x_right_pair=pair_spacing_m,
        )
        self.solver.walls = {
            "front_wall": WallSegment(
                name="front_wall",
                orientation="vertical",
                const_value=3.2,
                min_axis=-2.4,
                max_axis=0.0,
            ),
            "left_wall": WallSegment(
                name="left_wall",
                orientation="horizontal",
                const_value=0.0,
                min_axis=0.0,
                max_axis=3.2,
            ),
        }
        self.solver.wall_pairs = {
            "standard_pair": WallPair(
                name="standard_pair",
                x_wall_name="front_wall",
                x_wall_role="front",
                side_wall_name="left_wall",
                side_wall_role="left",
                corner_x=3.2,
                corner_y=0.0,
                corner_yaw_deg=0.0,
            )
        }

    def _make_scene_profile(self, *, xy_yaw_source: str | None = None) -> dict:
        region = {
            "name": "standard_region",
            "x_range": [0.0, 0.8],
            "y_range": [-1.8, -1.0],
            "yaw_deg": 0.0,
            "active_wall_pair": "standard_pair",
            "priority": 200,
        }
        if xy_yaw_source is not None:
            region["xy_yaw_source"] = xy_yaw_source
        return {
            "wall_selector": {
                "yaw_tolerance_deg": 10.0,
                "regions": [region],
            },
            "solver": {
                "min_valid_corner_beams": 3,
                "max_theta_abs_deg": 45.0,
                "wall_hit_tolerance_m": 1e-6,
                "wall_extent_margin_m": 1e-6,
                "max_correction_xy_m": 0.15,
                "max_correction_yaw_deg": 10.0,
                "residual_thresh_m": 0.03,
            },
        }

    def _make_range_frame(self, **ranges: float) -> RangeFrame:
        frame_ranges = {name: 0.0 for name in SENSOR_ORDER}
        frame_valid = {name: False for name in SENSOR_ORDER}
        for name, value in ranges.items():
            frame_ranges[name] = value
            frame_valid[name] = True
        return RangeFrame(
            stamp=Time(nanoseconds=0),
            ranges=frame_ranges,
            valid=frame_valid,
        )

    def test_refine_closed_form_defaults_to_side_laser_yaw(self) -> None:
        self.solver.active_scene = "mode_red"
        self.solver.scene_profiles = {
            "mode_red": self._make_scene_profile(),
        }
        coarse = CoarsePose(
            stamp=Time(nanoseconds=0),
            x=0.6,
            y=-1.125,
            z=0.0,
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_deg=8.0,
        )
        range_frame = self._make_range_frame(
            front_center=2.6,
            left_front=1.125,
            left_rear=1.125,
        )

        result = self.solver.refine(coarse, range_frame, Time(nanoseconds=0))

        self.assertEqual(result.state, STATE_REFINED)
        self.assertAlmostEqual(result.x, 0.6, places=6)
        self.assertAlmostEqual(result.y, -1.125, places=6)
        self.assertAlmostEqual(result.yaw_deg, 0.0, places=6)
        solver_debug = result.debug["solver_debug"]
        self.assertEqual(solver_debug["xy_yaw_source"], "side_laser")
        self.assertAlmostEqual(
            solver_debug["side_laser_yaw_in_corner_deg"], 0.0, places=6
        )
        self.assertAlmostEqual(solver_debug["lidar_yaw_in_corner_deg"], 8.0, places=6)
        self.assertAlmostEqual(
            solver_debug["xy_projection_yaw_in_corner_deg"], 0.0, places=6
        )

    def test_refine_closed_form_can_use_lidar_yaw_for_xy(self) -> None:
        self.solver.active_scene = "mode_red"
        self.solver.scene_profiles = {
            "mode_red": self._make_scene_profile(xy_yaw_source="lidar"),
        }
        coarse = CoarsePose(
            stamp=Time(nanoseconds=0),
            x=0.6,
            y=-1.125,
            z=0.0,
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_deg=8.0,
        )
        range_frame = self._make_range_frame(
            front_center=2.6,
            left_front=1.125,
            left_rear=1.125,
        )

        result = self.solver.refine(coarse, range_frame, Time(nanoseconds=0))

        expected_cos = math.cos(math.radians(8.0))
        self.assertEqual(result.state, STATE_REFINED)
        self.assertAlmostEqual(result.x, 3.2 - 2.6 * expected_cos, places=6)
        self.assertAlmostEqual(result.y, -1.125 * expected_cos, places=6)
        self.assertAlmostEqual(result.yaw_deg, 8.0, places=6)
        self.assertAlmostEqual(
            result.publish_x, 3.2 - 2.6 * expected_cos, places=6
        )
        self.assertAlmostEqual(result.publish_y, -1.125 * expected_cos, places=6)
        self.assertAlmostEqual(result.publish_z, 0.0, places=6)
        self.assertIsNone(result.publish_yaw_deg)
        solver_debug = result.debug["solver_debug"]
        self.assertEqual(solver_debug["xy_yaw_source"], "lidar")
        self.assertAlmostEqual(
            solver_debug["side_laser_yaw_in_corner_deg"], 0.0, places=6
        )
        self.assertAlmostEqual(solver_debug["lidar_yaw_in_corner_deg"], 8.0, places=6)
        self.assertAlmostEqual(
            solver_debug["xy_projection_yaw_in_corner_deg"], 8.0, places=6
        )

    def test_refine_rejects_stale_range_frame(self) -> None:
        self.solver.active_scene = "mode_red"
        self.solver.scene_profiles = {
            "mode_red": self._make_scene_profile(),
        }
        coarse = CoarsePose(
            stamp=Time(nanoseconds=0),
            x=0.6,
            y=-1.125,
            z=0.0,
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_deg=0.0,
        )
        range_frame = self._make_range_frame(front_center=2.6)

        result = self.solver.refine(
            coarse,
            range_frame,
            Time(nanoseconds=300_000_000),
            max_range_frame_age_ms=250.0,
        )

        self.assertEqual(result.state, STATE_COARSE_ONLY)
        self.assertEqual(result.reason, "STALE_SERIAL_RANGE_FRAME")
        self.assertEqual(result.pose_source, "lidar_coarse")
        self.assertAlmostEqual(result.prior_age_ms, 300.0, places=6)


class TestCompensatedFrontSideSolver(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = PoseSolveLayer.__new__(PoseSolveLayer)
        pair_spacing_m = 0.45816
        half_pair_spacing_m = pair_spacing_m * 0.5
        zero_mount = dict(min_range_m=0.01, max_range_m=20.0)
        self.solver.sensor_mounts = {
            "front_center": SensorMount(
                pos_x=0.0,
                pos_y=0.0,
                dir_x=1.0,
                dir_y=0.0,
                **zero_mount,
            ),
            "rear_center": SensorMount(
                pos_x=0.0,
                pos_y=0.0,
                dir_x=-1.0,
                dir_y=0.0,
                **zero_mount,
            ),
            "left_front": SensorMount(
                pos_x=half_pair_spacing_m,
                pos_y=0.0,
                dir_x=0.0,
                dir_y=1.0,
                **zero_mount,
            ),
            "left_rear": SensorMount(
                pos_x=-half_pair_spacing_m,
                pos_y=0.0,
                dir_x=0.0,
                dir_y=1.0,
                **zero_mount,
            ),
            "right_front": SensorMount(
                pos_x=half_pair_spacing_m,
                pos_y=0.0,
                dir_x=0.0,
                dir_y=-1.0,
                **zero_mount,
            ),
            "right_rear": SensorMount(
                pos_x=-half_pair_spacing_m,
                pos_y=0.0,
                dir_x=0.0,
                dir_y=-1.0,
                **zero_mount,
            ),
        }
        self.solver.sensor_geometry = SensorGeometry(
            x_front=0.0,
            x_rear=0.0,
            y_left=0.0,
            y_right=0.0,
            x_left_pair=pair_spacing_m,
            x_right_pair=pair_spacing_m,
        )
        self.solver.walls = {
            "red_special_front_vertical": WallSegment(
                name="red_special_front_vertical",
                orientation="vertical",
                const_value=3.2,
                min_axis=-2.4,
                max_axis=0.0,
            ),
            "red_special_side_horizontal": WallSegment(
                name="red_special_side_horizontal",
                orientation="horizontal",
                const_value=0.0,
                min_axis=0.0,
                max_axis=3.2,
            ),
            "blue_special_front_vertical": WallSegment(
                name="blue_special_front_vertical",
                orientation="vertical",
                const_value=3.2,
                min_axis=0.0,
                max_axis=2.4,
            ),
            "blue_special_side_horizontal": WallSegment(
                name="blue_special_side_horizontal",
                orientation="horizontal",
                const_value=0.0,
                min_axis=0.0,
                max_axis=3.2,
            ),
            "projected_local_front_horizontal": WallSegment(
                name="projected_local_front_horizontal",
                orientation="horizontal",
                const_value=0.0,
                min_axis=0.0,
                max_axis=3.2,
            ),
            "projected_local_side_vertical": WallSegment(
                name="projected_local_side_vertical",
                orientation="vertical",
                const_value=3.2,
                min_axis=-2.4,
                max_axis=0.0,
            ),
        }
        self.solver.wall_pairs = {
            "red_special_front_compensated": WallPair(
                name="red_special_front_compensated",
                x_wall_name="red_special_front_vertical",
                x_wall_role="front",
                side_wall_name="red_special_side_horizontal",
                side_wall_role="left",
                corner_x=3.2,
                corner_y=0.0,
                corner_yaw_deg=0.0,
            ),
            "blue_special_front_compensated": WallPair(
                name="blue_special_front_compensated",
                x_wall_name="blue_special_front_vertical",
                x_wall_role="front",
                side_wall_name="blue_special_side_horizontal",
                side_wall_role="right",
                corner_x=3.2,
                corner_y=0.0,
                corner_yaw_deg=0.0,
            ),
            "projected_local_rotated_pair": WallPair(
                name="projected_local_rotated_pair",
                x_wall_name="projected_local_front_horizontal",
                x_wall_role="front",
                side_wall_name="projected_local_side_vertical",
                side_wall_role="right",
                corner_x=3.2,
                corner_y=0.0,
                corner_yaw_deg=90.0,
            ),
        }

    def _make_scene_profile(
        self,
        *,
        pair_name: str,
        x_range: list[float],
        y_range: list[float],
        xy_yaw_source: str | None = None,
    ) -> dict:
        region = {
            "name": f"{pair_name}_region",
            "x_range": x_range,
            "y_range": y_range,
            "yaw_deg": 0.0,
            "active_wall_pair": pair_name,
            "special_solver": {
                "type": "compensated_front_side",
                "compensation_x0_m": 0.125,
                "max_iterations": 8,
                "theta_tolerance_deg": 0.01,
            },
            "priority": 200,
        }
        if xy_yaw_source is not None:
            region["xy_yaw_source"] = xy_yaw_source
        return {
            "wall_selector": {
                "yaw_tolerance_deg": 10.0,
                "regions": [region],
            },
            "solver": {
                "min_valid_corner_beams": 3,
                "max_theta_abs_deg": 45.0,
                "wall_hit_tolerance_m": 1e-6,
                "wall_extent_margin_m": 1e-6,
                "max_correction_xy_m": 0.15,
                "max_correction_yaw_deg": 10.0,
                "residual_thresh_m": 0.03,
            },
        }

    def _make_range_frame(self, **ranges: float) -> RangeFrame:
        frame_ranges = {name: 0.0 for name in SENSOR_ORDER}
        frame_valid = {name: False for name in SENSOR_ORDER}
        for name, value in ranges.items():
            frame_ranges[name] = value
            frame_valid[name] = True
        return RangeFrame(
            stamp=Time(nanoseconds=0),
            ranges=frame_ranges,
            valid=frame_valid,
        )

    def test_refine_uses_compensated_special_solver_for_red_region(self) -> None:
        self.solver.active_scene = "mode_red"
        self.solver.scene_profiles = {
            "mode_red": self._make_scene_profile(
                pair_name="red_special_front_compensated",
                x_range=[0.0, 0.8],
                y_range=[-1.8, -1.0],
            )
        }
        coarse = CoarsePose(
            stamp=Time(nanoseconds=0),
            x=0.6,
            y=-1.125,
            z=0.0,
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_deg=0.0,
        )
        range_frame = self._make_range_frame(
            front_center=2.6,
            left_front=1.0,
            left_rear=1.125,
        )

        result = self.solver.refine(coarse, range_frame, Time(nanoseconds=0))

        self.assertEqual(result.state, STATE_REFINED)
        self.assertEqual(result.pose_source, "compensated_front_side_corner_solver")
        self.assertAlmostEqual(result.x, 0.6, places=6)
        self.assertAlmostEqual(result.y, -1.125, places=6)
        self.assertAlmostEqual(result.yaw_deg, 0.0, places=6)
        self.assertAlmostEqual(result.residual_m, 0.0, places=6)
        special_debug = result.debug["solver_debug"]["special_solver_debug"]
        self.assertAlmostEqual(
            special_debug["corrected_side_front_range_m"], 1.125, places=6
        )
        self.assertTrue(special_debug["theta_converged"])

    def test_refine_compensated_special_solver_can_use_lidar_yaw(self) -> None:
        self.solver.active_scene = "mode_red"
        self.solver.scene_profiles = {
            "mode_red": self._make_scene_profile(
                pair_name="red_special_front_compensated",
                x_range=[0.0, 0.8],
                y_range=[-1.8, -1.0],
                xy_yaw_source="lidar",
            )
        }
        coarse = CoarsePose(
            stamp=Time(nanoseconds=0),
            x=0.6,
            y=-1.125,
            z=0.0,
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_deg=8.0,
        )
        range_frame = self._make_range_frame(
            front_center=2.6,
            left_front=1.0,
            left_rear=1.125,
        )

        result = self.solver.refine(coarse, range_frame, Time(nanoseconds=0))

        expected_cos = math.cos(math.radians(8.0))
        expected_corrected_side_front = 1.0 + 0.125 / expected_cos
        self.assertEqual(result.state, STATE_REFINED)
        self.assertEqual(result.pose_source, "compensated_front_side_corner_solver")
        self.assertAlmostEqual(result.x, 3.2 - 2.6 * expected_cos, places=6)
        self.assertAlmostEqual(
            result.y,
            -0.5 * (expected_corrected_side_front + 1.125) * expected_cos,
            places=6,
        )
        self.assertAlmostEqual(result.yaw_deg, 8.0, places=6)
        self.assertAlmostEqual(
            result.publish_x, 3.2 - 2.6 * expected_cos, places=6
        )
        self.assertAlmostEqual(
            result.publish_y,
            -0.5 * (expected_corrected_side_front + 1.125) * expected_cos,
            places=6,
        )
        self.assertAlmostEqual(result.publish_z, 0.0, places=6)
        self.assertIsNone(result.publish_yaw_deg)
        solver_debug = result.debug["solver_debug"]
        self.assertEqual(solver_debug["xy_yaw_source"], "lidar")
        self.assertAlmostEqual(solver_debug["lidar_yaw_in_corner_deg"], 8.0, places=6)
        self.assertAlmostEqual(
            solver_debug["xy_projection_yaw_in_corner_deg"], 8.0, places=6
        )
        special_debug = solver_debug["special_solver_debug"]
        self.assertEqual(special_debug["theta_iterations"], 0)
        self.assertTrue(special_debug["theta_converged"])
        self.assertAlmostEqual(
            special_debug["corrected_side_front_range_m"],
            expected_corrected_side_front,
            places=6,
        )

    def test_refine_uses_compensated_special_solver_for_blue_region(self) -> None:
        self.solver.active_scene = "mode_blue"
        self.solver.scene_profiles = {
            "mode_blue": self._make_scene_profile(
                pair_name="blue_special_front_compensated",
                x_range=[0.0, 0.8],
                y_range=[1.0, 1.8],
            )
        }
        coarse = CoarsePose(
            stamp=Time(nanoseconds=0),
            x=0.6,
            y=1.125,
            z=0.0,
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_deg=0.0,
        )
        range_frame = self._make_range_frame(
            front_center=2.6,
            right_front=1.0,
            right_rear=1.125,
        )

        result = self.solver.refine(coarse, range_frame, Time(nanoseconds=0))

        self.assertEqual(result.state, STATE_REFINED)
        self.assertEqual(result.pose_source, "compensated_front_side_corner_solver")
        self.assertAlmostEqual(result.x, 0.6, places=6)
        self.assertAlmostEqual(result.y, 1.125, places=6)
        self.assertAlmostEqual(result.yaw_deg, 0.0, places=6)
        self.assertAlmostEqual(result.residual_m, 0.0, places=6)
        special_debug = result.debug["solver_debug"]["special_solver_debug"]
        self.assertAlmostEqual(
            special_debug["corrected_side_front_range_m"], 1.125, places=6
        )
        self.assertTrue(special_debug["theta_converged"])

    def test_refine_uses_projected_xy_solver_for_red_region(self) -> None:
        self.solver.active_scene = "mode_red"
        self.solver.scene_profiles = {
            "mode_red": {
                "wall_selector": {
                    "yaw_tolerance_deg": 10.0,
                    "regions": [
                        {
                            "name": "rotated_projected_xy_region",
                            "x_range": [2.4, 2.8],
                            "y_range": [-1.4, -1.0],
                            "yaw_deg": 90.0,
                            "active_wall_pair": "projected_local_rotated_pair",
                            "special_solver": {
                                "type": "projected_xy_with_lidar_yaw",
                                "solve_axes": ["x", "y"],
                                "x_beam": "front_center",
                                "y_beam": "right_front",
                                "min_dir_component_abs": 0.2,
                                "max_x_correction_m": 0.2,
                                "max_y_correction_m": 0.2,
                            },
                            "priority": 200,
                        }
                    ],
                },
                "solver": {
                    "min_valid_corner_beams": 3,
                    "max_theta_abs_deg": 45.0,
                    "wall_hit_tolerance_m": 1e-6,
                    "wall_extent_margin_m": 1e-6,
                    "max_correction_xy_m": 0.15,
                    "max_correction_yaw_deg": 10.0,
                    "residual_thresh_m": 0.03,
                },
            }
        }
        coarse = CoarsePose(
            stamp=Time(nanoseconds=0),
            x=2.6,
            y=-1.125,
            z=0.0,
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_deg=90.0,
        )
        range_frame = self._make_range_frame(
            front_center=1.125,
            right_front=0.6,
        )

        result = self.solver.refine(coarse, range_frame, Time(nanoseconds=0))

        self.assertEqual(result.state, STATE_REFINED)
        self.assertEqual(result.pose_source, "projected_xy_with_lidar_yaw")
        self.assertAlmostEqual(result.x, 2.6, places=6)
        self.assertAlmostEqual(result.y, -1.125, places=6)
        self.assertAlmostEqual(result.yaw_deg, 90.0, places=6)
        self.assertAlmostEqual(result.residual_m, 0.0, places=6)
        self.assertEqual(result.selected_beams, ["front_center", "right_front"])
        self.assertAlmostEqual(result.publish_x, 2.6, places=6)
        self.assertAlmostEqual(result.publish_y, -1.125, places=6)
        self.assertAlmostEqual(result.publish_z, 0.0, places=6)
        self.assertIsNone(result.publish_yaw_deg)
        special_debug = result.debug["solver_debug"]["special_solver_debug"]
        self.assertEqual(special_debug["solver_type"], "projected_xy_with_lidar_yaw")
        self.assertEqual(special_debug["x_solver"]["beam_name"], "front_center")
        self.assertEqual(special_debug["y_solver"]["beam_name"], "right_front")

    def test_refine_projected_x_only_publishes_only_x(self) -> None:
        self.solver.active_scene = "mode_red"
        self.solver.scene_profiles = {
            "mode_red": {
                "wall_selector": {
                    "yaw_tolerance_deg": 10.0,
                    "regions": [
                        {
                            "name": "rotated_projected_x_region",
                            "x_range": [2.4, 2.8],
                            "y_range": [-1.4, -1.0],
                            "yaw_deg": 90.0,
                            "active_wall_pair": "projected_local_rotated_pair",
                            "special_solver": {
                                "type": "projected_xy_with_lidar_yaw",
                                "solve_axes": ["x"],
                                "x_beam": "front_center",
                                "min_dir_component_abs": 0.2,
                                "max_x_correction_m": 0.2,
                            },
                            "priority": 200,
                        }
                    ],
                },
                "solver": {
                    "min_valid_corner_beams": 3,
                    "max_theta_abs_deg": 45.0,
                    "wall_hit_tolerance_m": 1e-6,
                    "wall_extent_margin_m": 1e-6,
                    "max_correction_xy_m": 0.15,
                    "max_correction_yaw_deg": 10.0,
                    "residual_thresh_m": 0.03,
                },
            }
        }
        coarse = CoarsePose(
            stamp=Time(nanoseconds=0),
            x=2.6,
            y=-1.125,
            z=0.0,
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_deg=90.0,
        )
        range_frame = self._make_range_frame(front_center=1.125)

        result = self.solver.refine(coarse, range_frame, Time(nanoseconds=0))

        self.assertEqual(result.state, STATE_REFINED)
        self.assertAlmostEqual(result.publish_x, 2.6, places=6)
        self.assertIsNone(result.publish_y)
        self.assertIsNone(result.publish_z)
        self.assertIsNone(result.publish_yaw_deg)

    def test_refine_projected_y_only_publishes_only_y(self) -> None:
        self.solver.active_scene = "mode_red"
        self.solver.scene_profiles = {
            "mode_red": {
                "wall_selector": {
                    "yaw_tolerance_deg": 10.0,
                    "regions": [
                        {
                            "name": "rotated_projected_y_region",
                            "x_range": [2.4, 2.8],
                            "y_range": [-1.4, -1.0],
                            "yaw_deg": 90.0,
                            "active_wall_pair": "projected_local_rotated_pair",
                            "special_solver": {
                                "type": "projected_xy_with_lidar_yaw",
                                "solve_axes": ["y"],
                                "y_beam": "right_front",
                                "min_dir_component_abs": 0.2,
                                "max_y_correction_m": 0.2,
                            },
                            "priority": 200,
                        }
                    ],
                },
                "solver": {
                    "min_valid_corner_beams": 3,
                    "max_theta_abs_deg": 45.0,
                    "wall_hit_tolerance_m": 1e-6,
                    "wall_extent_margin_m": 1e-6,
                    "max_correction_xy_m": 0.15,
                    "max_correction_yaw_deg": 10.0,
                    "residual_thresh_m": 0.03,
                },
            }
        }
        coarse = CoarsePose(
            stamp=Time(nanoseconds=0),
            x=2.6,
            y=-1.125,
            z=0.0,
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_deg=90.0,
        )
        range_frame = self._make_range_frame(right_front=0.6)

        result = self.solver.refine(coarse, range_frame, Time(nanoseconds=0))

        self.assertEqual(result.state, STATE_REFINED)
        self.assertIsNone(result.publish_x)
        self.assertAlmostEqual(result.publish_y, -1.125, places=6)
        self.assertIsNone(result.publish_z)
        self.assertIsNone(result.publish_yaw_deg)

    def test_refine_projected_xy_solver_rejects_large_correction(self) -> None:
        self.solver.active_scene = "mode_red"
        self.solver.scene_profiles = {
            "mode_red": {
                "wall_selector": {
                    "yaw_tolerance_deg": 10.0,
                    "regions": [
                        {
                            "name": "rotated_projected_y_region",
                            "x_range": [2.2, 2.6],
                            "y_range": [-1.4, -1.0],
                            "yaw_deg": 90.0,
                            "active_wall_pair": "projected_local_rotated_pair",
                            "special_solver": {
                                "type": "projected_xy_with_lidar_yaw",
                                "solve_axes": ["y"],
                                "y_beam": "right_front",
                                "min_dir_component_abs": 0.2,
                                "max_y_correction_m": 0.2,
                            },
                            "priority": 200,
                        }
                    ],
                },
                "solver": {
                    "min_valid_corner_beams": 3,
                    "max_theta_abs_deg": 45.0,
                    "wall_hit_tolerance_m": 1e-6,
                    "wall_extent_margin_m": 1e-6,
                    "max_correction_xy_m": 0.15,
                    "max_correction_yaw_deg": 10.0,
                    "residual_thresh_m": 0.03,
                },
            }
        }
        coarse = CoarsePose(
            stamp=Time(nanoseconds=0),
            x=2.3,
            y=-1.125,
            z=0.0,
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_deg=90.0,
        )
        range_frame = self._make_range_frame(right_front=0.6)

        result = self.solver.refine(coarse, range_frame, Time(nanoseconds=0))

        self.assertEqual(result.state, STATE_COARSE_ONLY)
        self.assertEqual(result.reason, "PROJECTED_AXIS_CORRECTION_EXCEEDS_LIMIT")
        self.assertAlmostEqual(result.x, 2.3, places=6)
        special_debug = result.debug["solver_debug"]["special_solver_debug"]
        self.assertEqual(
            special_debug["failure_reason"],
            "PROJECTED_Y_CORRECTION_EXCEEDS_LIMIT",
        )

    def test_dual_wall_pair_front_fallback_supports_horizontal_front_wall(self) -> None:
        coarse = CoarsePose(
            stamp=Time(nanoseconds=0),
            x=2.3,
            y=-1.225,
            z=0.0,
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_deg=90.0,
        )
        range_frame = self._make_range_frame(front_center=1.125)
        wall_pairs = [
            WallPair(
                name="rotated_pair_left",
                x_wall_name="projected_local_front_horizontal",
                x_wall_role="front",
                side_wall_name="projected_local_side_vertical",
                side_wall_role="left",
                corner_x=3.2,
                corner_y=0.0,
                corner_yaw_deg=90.0,
            ),
            WallPair(
                name="rotated_pair_right",
                x_wall_name="projected_local_front_horizontal",
                x_wall_role="front",
                side_wall_name="projected_local_side_vertical",
                side_wall_role="right",
                corner_x=3.2,
                corner_y=0.0,
                corner_yaw_deg=90.0,
            ),
        ]

        result = self.solver._build_dual_wall_pair_front_x_fallback_result(
            coarse=coarse,
            range_frame=range_frame,
            wall_pairs=wall_pairs,
            solver_cfg={
                "max_correction_xy_m": 0.15,
                "max_correction_yaw_deg": 10.0,
            },
            prior_age_ms=0.0,
            region_name="rotated_dual_region",
        )

        self.assertEqual(result.state, STATE_REFINED)
        self.assertEqual(result.reason, "OK")
        self.assertAlmostEqual(result.x, 2.3, places=6)
        self.assertAlmostEqual(result.y, -1.125, places=6)
        self.assertAlmostEqual(result.yaw_deg, 90.0, places=6)
        self.assertEqual(result.selected_beams, ["front_center"])
        solver_debug = result.debug["solver_debug"]
        self.assertEqual(solver_debug["front_wall_orientation"], "horizontal")
        self.assertEqual(solver_debug["fallback_solved_axis"], "y")
        self.assertAlmostEqual(solver_debug["front_wall_const_y_m"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
