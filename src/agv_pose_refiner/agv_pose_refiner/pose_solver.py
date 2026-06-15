from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from rclpy.time import Time

from .common import (
    SENSOR_ORDER,
    STATE_CANNOT_LOCALIZE,
    STATE_COARSE_ONLY,
    STATE_REFINED,
    BeamSelection,
    CoarsePose,
    RangeFrame,
    RegionMatch,
    SensorGeometry,
    SensorMount,
    SolveResult,
    WallPair,
    WallSegment,
    rotate_2d,
    transform_pose_2d,
    time_diff_ms,
    wrap_deg,
)


class PoseSolveLayer:
    def __init__(
        self,
        logger: Any,
        sensors_config: Dict[str, Any],
        solver_config: Dict[str, Any],
        timing_cfg: Dict[str, Any],
    ) -> None:
        self._logger = logger
        self.sensor_mounts = self._parse_sensor_mounts(sensors_config)
        self.sensor_geometry = self._derive_sensor_geometry(self.sensor_mounts)

        self.world_frame_id = solver_config.get("world", {}).get("frame_id", "map")
        self.active_scene = solver_config.get("scene_manager", {}).get("active_scene")
        self.scene_profiles = solver_config.get("scene_profiles", {})
        if self.active_scene not in self.scene_profiles:
            raise RuntimeError(
                f"active_scene '{self.active_scene}' not found in scene_profiles"
            )

        self.walls = self._build_walls(solver_config)
        self.wall_pairs = self._build_wall_pairs(solver_config)
        self._validate_scene_profiles(self.scene_profiles)

    def refine(
        self,
        coarse: CoarsePose,
        range_frames: List[RangeFrame],
        now: Time,
    ) -> SolveResult:
        if not self._is_finite_pose(coarse):
            return self._make_result(
                state=STATE_CANNOT_LOCALIZE,
                reason="NON_FINITE_LIDAR_POSE",
                pose_source="invalid_input",
                debug=self._build_debug_payload(
                    coarse=coarse,
                    transport_delay_ms=None,
                    range_frame_found=None,
                    prior_age_ms=None,
                    range_frame_count=len(range_frames),
                    region_debug=None,
                ),
            )

        scene_cfg = self.scene_profiles.get(self.active_scene, {})
        region_match, region_debug = self._select_region_match_with_debug(
            scene_cfg, coarse
        )
        transport_delay_ms = time_diff_ms(now, coarse.stamp)
        range_frame = self._latest_range_frame(range_frames)
        if range_frame is None:
            return self._make_coarse_result(
                coarse,
                reason="NO_SERIAL_RANGE_AVAILABLE",
                debug=self._build_debug_payload(
                    coarse=coarse,
                    transport_delay_ms=transport_delay_ms,
                    range_frame_found=False,
                    prior_age_ms=None,
                    range_frame_count=len(range_frames),
                    region_debug=region_debug,
                ),
            )

        prior_age_ms = max(0.0, time_diff_ms(now, range_frame.stamp))
        usable_sensor_count = self._count_usable_sensors(range_frame)
        if region_match is None or region_match.wall_pair is None:
            return self._make_coarse_result(
                coarse,
                reason="NO_REGION_MATCHED",
                prior_age_ms=prior_age_ms,
                usable_sensor_count=usable_sensor_count,
                debug=self._build_debug_payload(
                    coarse=coarse,
                    transport_delay_ms=transport_delay_ms,
                    range_frame_found=True,
                    prior_age_ms=prior_age_ms,
                    range_frame_count=len(range_frames),
                    region_debug=region_debug,
                ),
            )

        solver_cfg = scene_cfg.get("solver", {})
        return self._solve_closed_form(
            coarse=coarse,
            range_frame=range_frame,
            wall_pair=region_match.wall_pair,
            solver_cfg=solver_cfg,
            prior_age_ms=prior_age_ms,
            region_name=region_match.name,
            usable_sensor_count=usable_sensor_count,
            debug=self._build_debug_payload(
                coarse=coarse,
                transport_delay_ms=transport_delay_ms,
                range_frame_found=True,
                prior_age_ms=prior_age_ms,
                range_frame_count=len(range_frames),
                region_debug=region_debug,
            ),
        )

    def _parse_sensor_mounts(self, config: Dict[str, Any]) -> Dict[str, SensorMount]:
        mounts_cfg = config.get("sensor_mounts", {})
        missing = [name for name in SENSOR_ORDER if name not in mounts_cfg]
        if missing:
            raise RuntimeError(f"sensor_mounts missing keys: {missing}")

        mounts: Dict[str, SensorMount] = {}
        for name in SENSOR_ORDER:
            item = mounts_cfg[name]
            pos = item.get("pos", [0.0, 0.0])
            direction = item.get("dir", [0.0, 0.0])
            norm = math.hypot(float(direction[0]), float(direction[1]))
            if norm < 1e-6:
                raise RuntimeError(f"Sensor {name} has zero direction vector")
            if abs(norm - 1.0) > 0.05:
                self._logger.warn(
                    f"Sensor {name} dir norm is {norm:.3f}, expected near 1.0"
                )
            mounts[name] = SensorMount(
                pos_x=float(pos[0]),
                pos_y=float(pos[1]),
                dir_x=float(direction[0]) / norm,
                dir_y=float(direction[1]) / norm,
                min_range_m=float(item.get("min_range_m", 0.03)),
                max_range_m=float(item.get("max_range_m", 2.0)),
            )
        return mounts

    def _derive_sensor_geometry(self, mounts: Dict[str, SensorMount]) -> SensorGeometry:
        return SensorGeometry(
            x_front=abs(mounts["front_center"].pos_x),
            x_rear=abs(mounts["rear_center"].pos_x),
            y_left=abs(mounts["left_front"].pos_y),
            y_right=abs(mounts["right_front"].pos_y),
            x_left_pair=abs(mounts["left_front"].pos_x - mounts["left_rear"].pos_x),
            x_right_pair=abs(mounts["right_front"].pos_x - mounts["right_rear"].pos_x),
        )

    def _build_walls(self, config: Dict[str, Any]) -> Dict[str, WallSegment]:
        corners_cfg = config.get("corner_pool", {})
        walls_cfg = config.get("wall_pool", {})
        walls: Dict[str, WallSegment] = {}
        for name, item in walls_cfg.items():
            if not item.get("enabled", True):
                continue
            p_from = corners_cfg[item["from"]]
            p_to = corners_cfg[item["to"]]
            x1, y1 = float(p_from[0]), float(p_from[1])
            x2, y2 = float(p_to[0]), float(p_to[1])
            if math.isclose(x1, x2, abs_tol=1e-9):
                walls[name] = WallSegment(
                    name=name,
                    orientation="vertical",
                    const_value=x1,
                    min_axis=min(y1, y2),
                    max_axis=max(y1, y2),
                )
            elif math.isclose(y1, y2, abs_tol=1e-9):
                walls[name] = WallSegment(
                    name=name,
                    orientation="horizontal",
                    const_value=y1,
                    min_axis=min(x1, x2),
                    max_axis=max(x1, x2),
                )
            else:
                raise RuntimeError(f"Wall {name} is not axis-aligned")
        return walls

    def _build_wall_pairs(self, config: Dict[str, Any]) -> Dict[str, WallPair]:
        pairs_cfg = config.get("wall_pair_candidates", {})
        pairs: Dict[str, WallPair] = {}
        for name, item in pairs_cfg.items():
            x_wall_cfg = item["x_wall"]
            side_wall_cfg = item["side_wall"]
            corner_pose = item["corner_world_pose"]
            pair = WallPair(
                name=name,
                x_wall_name=str(x_wall_cfg["wall"]),
                x_wall_role=str(x_wall_cfg["role"]),
                side_wall_name=str(side_wall_cfg["wall"]),
                side_wall_role=str(side_wall_cfg["role"]),
                corner_x=float(corner_pose["x"]),
                corner_y=float(corner_pose["y"]),
                corner_yaw_deg=float(corner_pose["yaw_deg"]),
            )
            self._validate_wall_pair(pair)
            pairs[name] = pair
        return pairs

    def _validate_wall_pair(self, pair: WallPair) -> None:
        if pair.x_wall_name not in self.walls:
            raise RuntimeError(f"Wall pair {pair.name} references missing x wall")
        if pair.side_wall_name not in self.walls:
            raise RuntimeError(f"Wall pair {pair.name} references missing side wall")
        if pair.x_wall_role not in {"front", "rear"}:
            raise RuntimeError(f"Wall pair {pair.name} x_wall.role must be front/rear")
        if pair.side_wall_role not in {"left", "right"}:
            raise RuntimeError(f"Wall pair {pair.name} side_wall.role must be left/right")
        x_wall = self.walls[pair.x_wall_name]
        side_wall = self.walls[pair.side_wall_name]
        if x_wall.orientation == side_wall.orientation:
            raise RuntimeError(f"Wall pair {pair.name} walls must be orthogonal")

    def _validate_scene_profiles(self, scene_profiles: Dict[str, Any]) -> None:
        for scene_name, scene_cfg in scene_profiles.items():
            selector = scene_cfg.get("wall_selector", {})
            default_yaw_tolerance_deg = float(selector.get("yaw_tolerance_deg", 15.0))
            if default_yaw_tolerance_deg < 0.0:
                raise RuntimeError(
                    f"scene_profiles.{scene_name}.wall_selector.yaw_tolerance_deg must be >= 0"
                )
            regions = selector.get("regions", [])
            if not isinstance(regions, list):
                raise RuntimeError(
                    f"scene_profiles.{scene_name}.wall_selector.regions must be a list"
                )
            for index, region in enumerate(regions):
                if not isinstance(region, dict):
                    raise RuntimeError(
                        f"scene_profiles.{scene_name}.wall_selector.regions[{index}] must be a mapping"
                    )
                has_xy_range = "x_range" in region or "y_range" in region
                has_xy_anchor = "x" in region or "y" in region
                if has_xy_range and has_xy_anchor:
                    raise RuntimeError(
                        "Each wall_selector region must use either x_range/y_range or x/y, not both"
                    )
                if has_xy_range:
                    x_range = region.get("x_range", [])
                    y_range = region.get("y_range", [])
                    if len(x_range) != 2 or len(y_range) != 2:
                        raise RuntimeError(
                            "Each range-based wall_selector region must define exactly one x_range and one y_range"
                        )
                    float(x_range[0])
                    float(x_range[1])
                    float(y_range[0])
                    float(y_range[1])
                elif has_xy_anchor:
                    if "x" not in region or "y" not in region:
                        raise RuntimeError(
                            "Each point-based wall_selector region must define both x and y"
                        )
                    float(region["x"])
                    float(region["y"])
                else:
                    raise RuntimeError(
                        "Each wall_selector region must define either x_range/y_range or x/y"
                    )

                pair_name = region.get("active_wall_pair")
                if pair_name is None or isinstance(pair_name, list):
                    raise RuntimeError(
                        "Each wall_selector region must bind exactly one active_wall_pair"
                    )
                if str(pair_name) not in self.wall_pairs:
                    raise RuntimeError(
                        f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                        f"references unknown active_wall_pair '{pair_name}'"
                    )
                if "yaw_deg" in region:
                    float(region["yaw_deg"])
                    yaw_tolerance_deg = float(
                        region.get("yaw_tolerance_deg", default_yaw_tolerance_deg)
                    )
                    if yaw_tolerance_deg < 0.0:
                        raise RuntimeError(
                            f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                            "yaw_tolerance_deg must be >= 0"
                        )
                elif "yaw_tolerance_deg" in region:
                    raise RuntimeError(
                        f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                        "cannot define yaw_tolerance_deg without yaw_deg"
                    )
                if "max_lidar_distance_to_center_m" in region:
                    max_distance_m = float(region["max_lidar_distance_to_center_m"])
                    if max_distance_m < 0.0:
                        raise RuntimeError(
                            f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                            "max_lidar_distance_to_center_m must be >= 0"
        )

    def _is_finite_pose(self, coarse: CoarsePose) -> bool:
        return (
            math.isfinite(coarse.x)
            and math.isfinite(coarse.y)
            and math.isfinite(coarse.z)
            and math.isfinite(coarse.roll_rad)
            and math.isfinite(coarse.pitch_rad)
            and math.isfinite(coarse.yaw_deg)
        )

    def _latest_range_frame(self, frames: List[RangeFrame]) -> Optional[RangeFrame]:
        if not frames:
            return None
        return frames[-1]

    def _select_region_match_with_debug(
        self, scene_cfg: Dict[str, Any], coarse: CoarsePose
    ) -> Tuple[Optional[RegionMatch], Dict[str, Any]]:
        selector = scene_cfg.get("wall_selector", {})
        regions = selector.get("regions", [])
        default_yaw_tolerance_deg = float(selector.get("yaw_tolerance_deg", 15.0))
        best_region: Optional[Dict[str, Any]] = None
        best_priority = -10**9
        best_position_score = float("inf")
        best_has_yaw_rule = False
        best_yaw_error_deg = float("inf")
        candidate_debug: List[Dict[str, Any]] = []
        for region in regions:
            position_match, position_score = self._match_region_position(region, coarse)
            yaw_match, yaw_error_deg = self._match_region_yaw(
                region, coarse, default_yaw_tolerance_deg
            )
            candidate_debug.append(
                self._build_region_candidate_debug(
                    region=region,
                    coarse=coarse,
                    position_match=position_match,
                    position_score=position_score,
                    yaw_match=yaw_match,
                    yaw_error_deg=yaw_error_deg,
                    default_yaw_tolerance_deg=default_yaw_tolerance_deg,
                )
            )
            if not position_match:
                continue
            if not yaw_match:
                continue
            priority = int(region.get("priority", 0))
            has_yaw_rule = "yaw_deg" in region
            if (
                priority > best_priority
                or (
                    priority == best_priority
                    and position_score < best_position_score - 1e-9
                )
                or (
                    priority == best_priority
                    and abs(position_score - best_position_score) <= 1e-9
                    and has_yaw_rule
                    and not best_has_yaw_rule
                )
                or (
                    priority == best_priority
                    and abs(position_score - best_position_score) <= 1e-9
                    and has_yaw_rule == best_has_yaw_rule
                    and yaw_error_deg < best_yaw_error_deg - 1e-9
                )
            ):
                best_region = region
                best_priority = priority
                best_position_score = position_score
                best_has_yaw_rule = has_yaw_rule
                best_yaw_error_deg = yaw_error_deg

        matched_region_name: Optional[str] = None
        matched_wall_pair_name: Optional[str] = None
        if best_region is None:
            return None, {
                "evaluated": True,
                "matched": False,
                "matched_region_name": None,
                "matched_wall_pair_name": None,
                "candidate_count": len(candidate_debug),
                "candidates": candidate_debug,
            }

        pair_name = best_region.get("active_wall_pair")
        if pair_name is None:
            return None, {
                "evaluated": True,
                "matched": False,
                "matched_region_name": str(best_region.get("name", "")),
                "matched_wall_pair_name": None,
                "candidate_count": len(candidate_debug),
                "candidates": candidate_debug,
            }
        wall_pair = self.wall_pairs.get(str(pair_name))
        matched_region_name = str(best_region.get("name", ""))
        matched_wall_pair_name = str(pair_name)
        for candidate in candidate_debug:
            if candidate.get("name") == matched_region_name:
                candidate["matched"] = True

        return RegionMatch(
            name=matched_region_name,
            wall_pair=wall_pair,
        ), {
            "evaluated": True,
            "matched": wall_pair is not None,
            "matched_region_name": matched_region_name,
            "matched_wall_pair_name": matched_wall_pair_name if wall_pair is not None else None,
            "candidate_count": len(candidate_debug),
            "candidates": candidate_debug,
        }

    def _build_region_candidate_debug(
        self,
        region: Dict[str, Any],
        coarse: CoarsePose,
        position_match: bool,
        position_score: float,
        yaw_match: bool,
        yaw_error_deg: float,
        default_yaw_tolerance_deg: float,
    ) -> Dict[str, Any]:
        candidate: Dict[str, Any] = {
            "name": str(region.get("name", "")),
            "priority": int(region.get("priority", 0)),
            "position_match": position_match,
            "yaw_match": yaw_match,
            "matched": False,
            "position_score_m": self._debug_float(position_score),
            "yaw_error_deg": self._debug_float(yaw_error_deg),
            "expected_yaw_deg": (
                float(region["yaw_deg"]) if "yaw_deg" in region else None
            ),
            "yaw_tolerance_deg": (
                float(region.get("yaw_tolerance_deg", default_yaw_tolerance_deg))
                if "yaw_deg" in region
                else None
            ),
        }
        if "x_range" in region or "y_range" in region:
            x_range = [float(value) for value in region.get("x_range", [])]
            y_range = [float(value) for value in region.get("y_range", [])]
            candidate["x_range"] = x_range
            candidate["y_range"] = y_range
            candidate["coarse_x_in_range"] = (
                len(x_range) == 2 and x_range[0] <= coarse.x <= x_range[1]
            )
            candidate["coarse_y_in_range"] = (
                len(y_range) == 2 and y_range[0] <= coarse.y <= y_range[1]
            )
        else:
            candidate["anchor_x"] = (
                float(region["x"]) if "x" in region else None
            )
            candidate["anchor_y"] = (
                float(region["y"]) if "y" in region else None
            )
            candidate["max_lidar_distance_to_center_m"] = (
                float(region["max_lidar_distance_to_center_m"])
                if "max_lidar_distance_to_center_m" in region
                else None
            )

        if not position_match:
            candidate["reject_reason"] = "POSITION_OUT_OF_RANGE"
        elif not yaw_match:
            candidate["reject_reason"] = "YAW_OUT_OF_RANGE"
        else:
            candidate["reject_reason"] = None
        return candidate

    def _match_region_position(
        self, region: Dict[str, Any], coarse: CoarsePose
    ) -> Tuple[bool, float]:
        if "x_range" in region or "y_range" in region:
            x_range = region.get("x_range", [])
            y_range = region.get("y_range", [])
            if len(x_range) != 2 or len(y_range) != 2:
                return False, float("inf")
            x_min = float(x_range[0])
            x_max = float(x_range[1])
            y_min = float(y_range[0])
            y_max = float(y_range[1])
            if not (x_min <= coarse.x <= x_max and y_min <= coarse.y <= y_max):
                return False, float("inf")
            center_x = 0.5 * (x_min + x_max)
            center_y = 0.5 * (y_min + y_max)
        else:
            if "x" not in region or "y" not in region:
                return False, float("inf")
            center_x = float(region["x"])
            center_y = float(region["y"])

        position_score = math.hypot(coarse.x - center_x, coarse.y - center_y)
        max_distance_m = region.get("max_lidar_distance_to_center_m")
        if max_distance_m is not None and position_score > float(max_distance_m):
            return False, float("inf")
        return True, position_score

    def _match_region_yaw(
        self,
        region: Dict[str, Any],
        coarse: CoarsePose,
        default_yaw_tolerance_deg: float,
    ) -> Tuple[bool, float]:
        if "yaw_deg" not in region:
            return True, 0.0
        yaw_error_deg = abs(wrap_deg(coarse.yaw_deg - float(region["yaw_deg"])))
        yaw_tolerance_deg = float(
            region.get("yaw_tolerance_deg", default_yaw_tolerance_deg)
        )
        return yaw_error_deg <= yaw_tolerance_deg, yaw_error_deg

    def _build_debug_payload(
        self,
        coarse: CoarsePose,
        transport_delay_ms: Optional[float],
        range_frame_found: Optional[bool],
        prior_age_ms: Optional[float],
        range_frame_count: int,
        region_debug: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "scene": self.active_scene,
            "coarse_pose": {
                "x": self._debug_float(coarse.x),
                "y": self._debug_float(coarse.y),
                "z": self._debug_float(coarse.z),
                "yaw_deg": self._debug_float(coarse.yaw_deg),
                "roll_rad": self._debug_float(coarse.roll_rad),
                "pitch_rad": self._debug_float(coarse.pitch_rad),
            },
            "timing_debug": {
                "transport_delay_ms": self._debug_float(transport_delay_ms),
                "range_frame_found": range_frame_found,
                "range_frame_age_ms": self._debug_float(prior_age_ms),
                "range_frame_count": int(range_frame_count),
                "range_frame_source": "latest_complete_serial_cycle",
                "lidar_timestamp_sync_enabled": False,
            },
            "region_debug": region_debug
            if region_debug is not None
            else {"evaluated": False, "matched": False},
        }

    def _debug_float(self, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if not math.isfinite(value):
            return None
        return float(value)

    def _make_result(
        self,
        state: str,
        reason: str,
        pose_source: str,
        *,
        localized: bool = False,
        x: Optional[float] = None,
        y: Optional[float] = None,
        yaw_deg: Optional[float] = None,
        valid_beam_count: int = 0,
        score: float = 0.0,
        prior_age_ms: Optional[float] = None,
        usable_sensor_count: int = 0,
        selected_beam_count: int = 0,
        selected_valid_beam_count: int = 0,
        target_hit_count: int = 0,
        debug: Optional[Dict[str, Any]] = None,
        residual_m: Optional[float] = None,
        wall_pair_name: Optional[str] = None,
        region_name: Optional[str] = None,
        beam_mode: Optional[str] = None,
        selected_beams: Optional[List[str]] = None,
        yaw_in_corner_deg: Optional[float] = None,
    ) -> SolveResult:
        return SolveResult(
            state=state,
            reason=reason,
            pose_source=pose_source,
            localized=localized,
            x=x,
            y=y,
            yaw_deg=yaw_deg,
            valid_beam_count=valid_beam_count,
            score=score,
            prior_age_ms=prior_age_ms,
            usable_sensor_count=usable_sensor_count,
            selected_beam_count=selected_beam_count,
            selected_valid_beam_count=selected_valid_beam_count,
            target_hit_count=target_hit_count,
            debug=debug,
            residual_m=residual_m,
            wall_pair_name=wall_pair_name,
            region_name=region_name,
            beam_mode=beam_mode,
            selected_beams=selected_beams,
            yaw_in_corner_deg=yaw_in_corner_deg,
        )

    def _make_coarse_result(
        self,
        coarse: CoarsePose,
        reason: str,
        *,
        valid_beam_count: int = 0,
        prior_age_ms: Optional[float] = None,
        usable_sensor_count: int = 0,
        selected_beam_count: int = 0,
        selected_valid_beam_count: int = 0,
        target_hit_count: int = 0,
        debug: Optional[Dict[str, Any]] = None,
        residual_m: Optional[float] = None,
        wall_pair_name: Optional[str] = None,
        region_name: Optional[str] = None,
        beam_mode: Optional[str] = None,
        selected_beams: Optional[List[str]] = None,
        yaw_in_corner_deg: Optional[float] = None,
    ) -> SolveResult:
        return self._make_result(
            state=STATE_COARSE_ONLY,
            reason=reason,
            pose_source="lidar_coarse",
            x=coarse.x,
            y=coarse.y,
            yaw_deg=coarse.yaw_deg,
            valid_beam_count=valid_beam_count,
            prior_age_ms=prior_age_ms,
            usable_sensor_count=usable_sensor_count,
            selected_beam_count=selected_beam_count,
            selected_valid_beam_count=selected_valid_beam_count,
            target_hit_count=target_hit_count,
            debug=debug,
            residual_m=residual_m,
            wall_pair_name=wall_pair_name,
            region_name=region_name,
            beam_mode=beam_mode,
            selected_beams=selected_beams,
            yaw_in_corner_deg=yaw_in_corner_deg,
        )

    def _make_solver_candidate_result(
        self,
        reason: str,
        *,
        pose_x: float,
        pose_y: float,
        pose_yaw_deg: float,
        valid_beam_count: int = 0,
        score: float = 0.0,
        prior_age_ms: Optional[float] = None,
        usable_sensor_count: int = 0,
        selected_beam_count: int = 0,
        selected_valid_beam_count: int = 0,
        target_hit_count: int = 0,
        debug: Optional[Dict[str, Any]] = None,
        residual_m: Optional[float] = None,
        wall_pair_name: Optional[str] = None,
        region_name: Optional[str] = None,
        beam_mode: Optional[str] = None,
        selected_beams: Optional[List[str]] = None,
        yaw_in_corner_deg: Optional[float] = None,
    ) -> SolveResult:
        return self._make_result(
            state=STATE_COARSE_ONLY,
            reason=reason,
            pose_source="closed_form_corner_solver_candidate",
            x=pose_x,
            y=pose_y,
            yaw_deg=pose_yaw_deg,
            valid_beam_count=valid_beam_count,
            score=score,
            prior_age_ms=prior_age_ms,
            usable_sensor_count=usable_sensor_count,
            selected_beam_count=selected_beam_count,
            selected_valid_beam_count=selected_valid_beam_count,
            target_hit_count=target_hit_count,
            debug=debug,
            residual_m=residual_m,
            wall_pair_name=wall_pair_name,
            region_name=region_name,
            beam_mode=beam_mode,
            selected_beams=selected_beams,
            yaw_in_corner_deg=yaw_in_corner_deg,
        )

    def _solve_closed_form(
        self,
        coarse: CoarsePose,
        range_frame: RangeFrame,
        wall_pair: WallPair,
        solver_cfg: Dict[str, Any],
        prior_age_ms: float,
        region_name: Optional[str] = None,
        usable_sensor_count: int = 0,
        debug: Optional[Dict[str, Any]] = None,
    ) -> SolveResult:
        beam_selection = self._select_beam_set_for_wall_pair(wall_pair, coarse.yaw_deg)
        selected_beams = beam_selection.required_beams()
        selected_beam_count = len(selected_beams)
        selected_valid_beam_count = sum(
            1 for name in selected_beams if range_frame.valid.get(name, False)
        )
        min_valid_corner_beams = int(solver_cfg.get("min_valid_corner_beams", 3))
        if selected_valid_beam_count < min_valid_corner_beams:
            return self._make_coarse_result(
                coarse,
                reason="INSUFFICIENT_VALID_BEAMS",
                valid_beam_count=selected_valid_beam_count,
                prior_age_ms=prior_age_ms,
                usable_sensor_count=usable_sensor_count,
                selected_beam_count=selected_beam_count,
                selected_valid_beam_count=selected_valid_beam_count,
                debug=debug,
                wall_pair_name=wall_pair.name,
                region_name=region_name,
                beam_mode=beam_selection.beam_mode,
                selected_beams=selected_beams,
                yaw_in_corner_deg=beam_selection.yaw_in_corner_deg,
            )

        r_x = range_frame.ranges[beam_selection.x_beam]
        r_sf = range_frame.ranges[beam_selection.side_front_beam]
        r_sr = range_frame.ranges[beam_selection.side_rear_beam]
        x_pair = beam_selection.pair_spacing_m
        theta_side_deg = math.degrees(math.atan2(r_sf - r_sr, x_pair))
        if beam_selection.side_beam_role == "left":
            theta_side_deg = -theta_side_deg

        max_theta_abs_deg = float(solver_cfg.get("max_theta_abs_deg", 45.0))
        debug = self._with_solver_debug(
            debug,
            beam_selection=beam_selection,
            theta_side_deg=theta_side_deg,
        )
        if abs(theta_side_deg) > max_theta_abs_deg:
            return self._make_coarse_result(
                coarse,
                reason="THETA_EXCEEDS_LIMIT",
                valid_beam_count=selected_valid_beam_count,
                prior_age_ms=prior_age_ms,
                usable_sensor_count=usable_sensor_count,
                selected_beam_count=selected_beam_count,
                selected_valid_beam_count=selected_valid_beam_count,
                debug=debug,
                wall_pair_name=wall_pair.name,
                region_name=region_name,
                beam_mode=beam_selection.beam_mode,
                selected_beams=selected_beams,
                yaw_in_corner_deg=beam_selection.yaw_in_corner_deg,
            )

        c = math.cos(math.radians(theta_side_deg))
        sx = -1.0 if beam_selection.x_beam_role == "front" else 1.0
        sy = 1.0 if beam_selection.side_beam_role == "right" else -1.0
        d_x = beam_selection.x_offset_m
        d_y = beam_selection.side_offset_m

        x_corner = sx * (r_x + d_x) * c
        y_corner = sy * (d_y + 0.5 * (r_sf + r_sr)) * c
        yaw_corner_deg = (
            theta_side_deg
            if beam_selection.x_beam_role == "front"
            else wrap_deg(theta_side_deg + 180.0)
        )

        # The closed-form solver produces pose in the corner-local frame first.
        # Rotate and translate it with the configured corner world pose.
        x_map, y_map, yaw_map_deg = transform_pose_2d(
            wall_pair.corner_x,
            wall_pair.corner_y,
            wall_pair.corner_yaw_deg,
            x_corner,
            y_corner,
            yaw_corner_deg,
        )

        max_correction_xy_m = float(solver_cfg.get("max_correction_xy_m", 0.15))
        max_correction_yaw_deg = float(solver_cfg.get("max_correction_yaw_deg", 10.0))
        delta_x = x_map - coarse.x
        delta_y = y_map - coarse.y
        delta_yaw_deg = wrap_deg(yaw_map_deg - coarse.yaw_deg)
        delta_xy_norm_m = math.hypot(delta_x, delta_y)
        debug = self._with_solver_debug(
            debug,
            beam_selection=beam_selection,
            theta_side_deg=theta_side_deg,
            corner_pose={
                "x": x_corner,
                "y": y_corner,
                "yaw_deg": yaw_corner_deg,
            },
            corner_world_pose={
                "x": wall_pair.corner_x,
                "y": wall_pair.corner_y,
                "yaw_deg": wall_pair.corner_yaw_deg,
            },
            candidate_pose={
                "x": x_map,
                "y": y_map,
                "yaw_deg": yaw_map_deg,
            },
            correction_debug={
                "delta_x_m": delta_x,
                "delta_y_m": delta_y,
                "delta_xy_norm_m": delta_xy_norm_m,
                "delta_yaw_deg": delta_yaw_deg,
                "max_correction_xy_m": max_correction_xy_m,
                "max_correction_yaw_deg": max_correction_yaw_deg,
            },
        )

        residual_m, target_hits = self._evaluate_solution(
            pose_x=x_map,
            pose_y=y_map,
            pose_yaw_deg=yaw_map_deg,
            range_frame=range_frame,
            wall_pair=wall_pair,
            beam_selection=beam_selection,
            solver_cfg=solver_cfg,
        )
        debug = self._with_solver_debug(
            debug,
            beam_selection=beam_selection,
            theta_side_deg=theta_side_deg,
            corner_pose={
                "x": x_corner,
                "y": y_corner,
                "yaw_deg": yaw_corner_deg,
            },
            corner_world_pose={
                "x": wall_pair.corner_x,
                "y": wall_pair.corner_y,
                "yaw_deg": wall_pair.corner_yaw_deg,
            },
            candidate_pose={
                "x": x_map,
                "y": y_map,
                "yaw_deg": yaw_map_deg,
            },
            correction_debug={
                "delta_x_m": delta_x,
                "delta_y_m": delta_y,
                "delta_xy_norm_m": delta_xy_norm_m,
                "delta_yaw_deg": delta_yaw_deg,
                "max_correction_xy_m": max_correction_xy_m,
                "max_correction_yaw_deg": max_correction_yaw_deg,
            },
            residual_debug={
                "mean_residual_m": residual_m,
                "target_hit_count": target_hits,
            },
        )
        residual_thresh_m = float(solver_cfg.get("residual_thresh_m", 0.03))
        score = max(0.0, 1.0 - residual_m / max(residual_thresh_m, 1e-6))
        debug = self._with_solver_debug(
            debug,
            beam_selection=beam_selection,
            theta_side_deg=theta_side_deg,
            residual_debug={
                "mean_residual_m": residual_m,
                "target_hit_count": target_hits,
                "residual_thresh_m": residual_thresh_m,
                "min_valid_corner_beams": float(min_valid_corner_beams),
                "validation_gates_block_pose": 0.0,
                "would_fail_target_hit_gate": (
                    1.0 if target_hits < min_valid_corner_beams else 0.0
                ),
                "would_fail_residual_gate": (
                    1.0 if residual_m > residual_thresh_m else 0.0
                ),
            },
        )

        return self._make_result(
            state=STATE_REFINED,
            reason="OK",
            pose_source="closed_form_corner_solver",
            localized=True,
            x=x_map,
            y=y_map,
            yaw_deg=yaw_map_deg,
            valid_beam_count=selected_valid_beam_count,
            score=score,
            prior_age_ms=prior_age_ms,
            usable_sensor_count=usable_sensor_count,
            selected_beam_count=selected_beam_count,
            selected_valid_beam_count=selected_valid_beam_count,
            target_hit_count=target_hits,
            debug=debug,
            residual_m=residual_m,
            wall_pair_name=wall_pair.name,
            region_name=region_name,
            beam_mode=beam_selection.beam_mode,
            selected_beams=selected_beams,
            yaw_in_corner_deg=yaw_corner_deg,
        )

    def _count_usable_sensors(self, range_frame: RangeFrame) -> int:
        return sum(1 for name in SENSOR_ORDER if range_frame.valid.get(name, False))

    def _with_solver_debug(
        self,
        debug: Optional[Dict[str, Any]],
        *,
        beam_selection: Optional[BeamSelection] = None,
        theta_side_deg: Optional[float] = None,
        corner_pose: Optional[Dict[str, float]] = None,
        corner_world_pose: Optional[Dict[str, float]] = None,
        candidate_pose: Optional[Dict[str, float]] = None,
        correction_debug: Optional[Dict[str, float]] = None,
        residual_debug: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        merged = dict(debug or {})
        solver_debug = dict(merged.get("solver_debug", {}))
        if beam_selection is not None:
            solver_debug["beam_mode"] = beam_selection.beam_mode
            solver_debug["x_beam"] = beam_selection.x_beam
            solver_debug["side_front_beam"] = beam_selection.side_front_beam
            solver_debug["side_rear_beam"] = beam_selection.side_rear_beam
        if theta_side_deg is not None:
            solver_debug["theta_side_deg"] = self._debug_float(theta_side_deg)
        if corner_pose is not None:
            solver_debug["corner_pose"] = {
                "x": self._debug_float(corner_pose.get("x")),
                "y": self._debug_float(corner_pose.get("y")),
                "yaw_deg": self._debug_float(corner_pose.get("yaw_deg")),
            }
        if corner_world_pose is not None:
            solver_debug["corner_world_pose"] = {
                "x": self._debug_float(corner_world_pose.get("x")),
                "y": self._debug_float(corner_world_pose.get("y")),
                "yaw_deg": self._debug_float(corner_world_pose.get("yaw_deg")),
            }
        if candidate_pose is not None:
            solver_debug["candidate_pose"] = {
                "x": self._debug_float(candidate_pose.get("x")),
                "y": self._debug_float(candidate_pose.get("y")),
                "yaw_deg": self._debug_float(candidate_pose.get("yaw_deg")),
            }
        if correction_debug is not None:
            solver_debug["correction_debug"] = {
                key: self._debug_float(value) for key, value in correction_debug.items()
            }
        if residual_debug is not None:
            solver_debug["residual_debug"] = {
                key: self._debug_float(value) for key, value in residual_debug.items()
            }
        merged["solver_debug"] = solver_debug
        return merged

    def _select_beam_set_for_wall_pair(
        self, wall_pair: WallPair, coarse_yaw_deg: float
    ) -> BeamSelection:
        yaw_in_corner_deg = wrap_deg(coarse_yaw_deg - wall_pair.corner_yaw_deg)
        use_front = wall_pair.x_wall_role == "front"
        use_left = wall_pair.side_wall_role == "left"

        if use_front:
            x_beam = "front_center"
            x_offset_m = self.sensor_geometry.x_front
        else:
            x_beam = "rear_center"
            x_offset_m = self.sensor_geometry.x_rear

        if use_left:
            side_front_beam = "left_front"
            side_rear_beam = "left_rear"
            side_offset_m = self.sensor_geometry.y_left
            pair_spacing_m = self.sensor_geometry.x_left_pair
            side_beam_role = "left"
        else:
            side_front_beam = "right_front"
            side_rear_beam = "right_rear"
            side_offset_m = self.sensor_geometry.y_right
            pair_spacing_m = self.sensor_geometry.x_right_pair
            side_beam_role = "right"

        beam_mode = f"{'front' if use_front else 'rear'}_{'left' if use_left else 'right'}"
        return BeamSelection(
            x_beam=x_beam,
            x_beam_role="front" if use_front else "rear",
            x_offset_m=x_offset_m,
            side_front_beam=side_front_beam,
            side_rear_beam=side_rear_beam,
            side_beam_role=side_beam_role,
            side_offset_m=side_offset_m,
            pair_spacing_m=max(pair_spacing_m, 1e-6),
            yaw_in_corner_deg=yaw_in_corner_deg,
            beam_mode=beam_mode,
        )

    def _evaluate_solution(
        self,
        pose_x: float,
        pose_y: float,
        pose_yaw_deg: float,
        range_frame: RangeFrame,
        wall_pair: WallPair,
        beam_selection: BeamSelection,
        solver_cfg: Dict[str, Any],
    ) -> Tuple[float, int]:
        wall_hit_tolerance_m = float(solver_cfg.get("wall_hit_tolerance_m", 0.05))
        wall_extent_margin_m = float(solver_cfg.get("wall_extent_margin_m", 0.08))
        x_wall = self.walls[wall_pair.x_wall_name]
        side_wall = self.walls[wall_pair.side_wall_name]

        checks = [
            (beam_selection.x_beam, x_wall, side_wall),
            (beam_selection.side_front_beam, side_wall, x_wall),
            (beam_selection.side_rear_beam, side_wall, x_wall),
        ]

        residuals: List[float] = []
        target_hits = 0
        for beam_name, target_wall, _other_wall in checks:
            hit_x, hit_y = self._beam_hit_point(
                pose_x=pose_x,
                pose_y=pose_y,
                pose_yaw_deg=pose_yaw_deg,
                sensor_name=beam_name,
                measured_range=range_frame.ranges[beam_name],
            )
            target_distance, target_in_extent = self._point_to_wall_distance(
                hit_x,
                hit_y,
                target_wall,
                wall_extent_margin_m,
            )
            if target_distance <= wall_hit_tolerance_m and target_in_extent:
                target_hits += 1
            residuals.append(target_distance)

        mean_residual = sum(residuals) / max(len(residuals), 1)
        return mean_residual, target_hits

    def _beam_hit_point(
        self,
        pose_x: float,
        pose_y: float,
        pose_yaw_deg: float,
        sensor_name: str,
        measured_range: float,
    ) -> Tuple[float, float]:
        mount = self.sensor_mounts[sensor_name]
        origin_dx, origin_dy = rotate_2d(mount.pos_x, mount.pos_y, pose_yaw_deg)
        dir_dx, dir_dy = rotate_2d(mount.dir_x, mount.dir_y, pose_yaw_deg)
        origin_x = pose_x + origin_dx
        origin_y = pose_y + origin_dy
        hit_x = origin_x + measured_range * dir_dx
        hit_y = origin_y + measured_range * dir_dy
        return hit_x, hit_y

    def _point_to_wall_distance(
        self,
        x: float,
        y: float,
        wall: WallSegment,
        extent_margin_m: float,
    ) -> Tuple[float, bool]:
        if wall.orientation == "vertical":
            distance = abs(x - wall.const_value)
            in_extent = wall.min_axis - extent_margin_m <= y <= wall.max_axis + extent_margin_m
        else:
            distance = abs(y - wall.const_value)
            in_extent = wall.min_axis - extent_margin_m <= x <= wall.max_axis + extent_margin_m
        return distance, in_extent
