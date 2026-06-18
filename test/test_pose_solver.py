from __future__ import annotations

import unittest

from _module_stubs import install_test_stubs

install_test_stubs()

from rclpy.time import Time

from agv_pose_refiner_py.common import CoarsePose, SolveResult, WallPair
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


if __name__ == "__main__":
    unittest.main()
