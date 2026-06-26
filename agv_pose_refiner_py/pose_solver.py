from __future__ import annotations

from dataclasses import replace
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

PROJECTED_ALLOWED_BEAMS = set(SENSOR_ORDER)


class PoseSolveLayer:
    def __init__(
        self,
        logger: Any,
        sensors_config: Dict[str, Any],
        solver_config: Dict[str, Any],
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
        range_frame: Optional[RangeFrame],
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
                    range_frame_found=range_frame is not None,
                    prior_age_ms=None,
                    range_frame_count=1 if range_frame is not None else 0,
                    region_debug=None,
                ),
            )

        scene_cfg = self.scene_profiles.get(self.active_scene, {})
        region_match, region_debug = self._select_region_match_with_debug(
            scene_cfg, coarse
        )
        transport_delay_ms = time_diff_ms(now, coarse.stamp)
        if range_frame is None:
            return self._make_coarse_result(
                coarse,
                reason="NO_SERIAL_RANGE_AVAILABLE",
                debug=self._build_debug_payload(
                    coarse=coarse,
                    transport_delay_ms=transport_delay_ms,
                    range_frame_found=False,
                    prior_age_ms=None,
                    range_frame_count=0,
                    region_debug=region_debug,
                ),
            )

        prior_age_ms = max(0.0, time_diff_ms(now, range_frame.stamp))
        usable_sensor_count = self._count_usable_sensors(range_frame)
        debug_payload = self._build_debug_payload(
            coarse=coarse,
            transport_delay_ms=transport_delay_ms,
            range_frame_found=True,
            prior_age_ms=prior_age_ms,
            range_frame_count=1,
            region_debug=region_debug,
        )
        if region_match is None or not region_match.wall_pairs:
            return self._make_coarse_result(
                coarse,
                reason="NO_REGION_MATCHED",
                prior_age_ms=prior_age_ms,
                usable_sensor_count=usable_sensor_count,
                debug=debug_payload,
            )

        solver_cfg = scene_cfg.get("solver", {})
        region_cfg = region_match.region_config or {}
        special_solver_cfg = region_cfg.get("special_solver")
        if isinstance(special_solver_cfg, dict):
            special_solver_type = str(special_solver_cfg.get("type", ""))
            if special_solver_type == "compensated_front_side":
                return self._solve_compensated_front_side_region(
                    coarse=coarse,
                    range_frame=range_frame,
                    wall_pair=region_match.wall_pairs[0],
                    solver_cfg=solver_cfg,
                    special_solver_cfg=special_solver_cfg,
                    region_cfg=region_cfg,
                    prior_age_ms=prior_age_ms,
                    region_name=region_match.name,
                    usable_sensor_count=usable_sensor_count,
                    debug=debug_payload,
                )
            if special_solver_type == "projected_xy_with_lidar_yaw":
                return self._solve_projected_xy_with_lidar_yaw(
                    coarse=coarse,
                    range_frame=range_frame,
                    wall_pair=region_match.wall_pairs[0],
                    solver_cfg=solver_cfg,
                    special_solver_cfg=special_solver_cfg,
                    prior_age_ms=prior_age_ms,
                    region_name=region_match.name,
                    usable_sensor_count=usable_sensor_count,
                    debug=debug_payload,
                )
            raise RuntimeError(f"Unsupported special_solver.type '{special_solver_type}'")
        if len(region_match.wall_pairs) == 1:
            return self._solve_closed_form(
                coarse=coarse,
                range_frame=range_frame,
                wall_pair=region_match.wall_pairs[0],
                solver_cfg=solver_cfg,
                region_cfg=region_cfg,
                prior_age_ms=prior_age_ms,
                region_name=region_match.name,
                usable_sensor_count=usable_sensor_count,
                debug=debug_payload,
            )
        return self._solve_dual_wall_pair_region(
            coarse=coarse,
            range_frame=range_frame,
            wall_pairs=region_match.wall_pairs,
            solver_cfg=solver_cfg,
            region_cfg=region_cfg,
            prior_age_ms=prior_age_ms,
            region_name=region_match.name,
            usable_sensor_count=usable_sensor_count,
            debug=debug_payload,
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

                pair_names = self._region_wall_pair_names(region)
                if "active_wall_pair" in region and "active_wall_pairs" in region:
                    raise RuntimeError(
                        f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                        "cannot define both active_wall_pair and active_wall_pairs"
                    )
                if not pair_names:
                    raise RuntimeError(
                        f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                        "must define active_wall_pair or exactly two active_wall_pairs"
                    )
                if len(pair_names) != len(set(pair_names)):
                    raise RuntimeError(
                        f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                        "active_wall_pair names must be unique"
                    )
                for pair_name in pair_names:
                    if pair_name not in self.wall_pairs:
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
                raw_xy_yaw_source = region.get("xy_yaw_source")
                if raw_xy_yaw_source is not None:
                    xy_yaw_source = self._region_xy_yaw_source(region)
                    if xy_yaw_source not in {"side_laser", "lidar"}:
                        raise RuntimeError(
                            f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                            "xy_yaw_source must be side_laser or lidar"
                        )
                if "max_lidar_distance_to_center_m" in region:
                    max_distance_m = float(region["max_lidar_distance_to_center_m"])
                    if max_distance_m < 0.0:
                        raise RuntimeError(
                            f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                            "max_lidar_distance_to_center_m must be >= 0"
                        )
                special_solver_cfg = region.get("special_solver")
                if special_solver_cfg is not None:
                    if not isinstance(special_solver_cfg, dict):
                        raise RuntimeError(
                            f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                            "special_solver must be a mapping"
                        )
                    solver_type = str(special_solver_cfg.get("type", ""))
                    if solver_type not in {
                        "compensated_front_side",
                        "projected_xy_with_lidar_yaw",
                    }:
                        raise RuntimeError(
                            f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                            "special_solver.type must be compensated_front_side or projected_xy_with_lidar_yaw"
                        )
                    if len(pair_names) != 1:
                        raise RuntimeError(
                            f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                            "special_solver requires exactly one active_wall_pair"
                        )
                    wall_pair = self.wall_pairs[pair_names[0]]
                    if solver_type == "compensated_front_side":
                        if wall_pair.x_wall_role != "front":
                            raise RuntimeError(
                                f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                                "special_solver requires x_wall.role=front"
                            )
                        compensation_x0_m = float(
                            special_solver_cfg.get("compensation_x0_m", 0.125)
                        )
                        if compensation_x0_m <= 0.0:
                            raise RuntimeError(
                                f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                                "special_solver.compensation_x0_m must be > 0"
                            )
                        max_iterations = int(special_solver_cfg.get("max_iterations", 8))
                        if max_iterations <= 0:
                            raise RuntimeError(
                                f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                                "special_solver.max_iterations must be > 0"
                            )
                        theta_tolerance_deg = float(
                            special_solver_cfg.get("theta_tolerance_deg", 0.01)
                        )
                        if theta_tolerance_deg <= 0.0:
                            raise RuntimeError(
                                f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                                "special_solver.theta_tolerance_deg must be > 0"
                            )
                    else:
                        if (
                            raw_xy_yaw_source is not None
                            and self._region_xy_yaw_source(region) != "lidar"
                        ):
                            raise RuntimeError(
                                f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                                "projected_xy_with_lidar_yaw regions must not set xy_yaw_source to side_laser"
                            )
                        solve_axes = self._projected_xy_solve_axes(special_solver_cfg)
                        if not solve_axes:
                            raise RuntimeError(
                                f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                                "special_solver.solve_axes must be a non-empty list containing x and/or y"
                            )
                        if len(solve_axes) != len(set(solve_axes)):
                            raise RuntimeError(
                                f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                                "special_solver.solve_axes entries must be unique"
                            )
                        invalid_axes = [
                            axis for axis in solve_axes if axis not in {"x", "y"}
                        ]
                        if invalid_axes:
                            raise RuntimeError(
                                f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                                "special_solver.solve_axes entries must be x or y"
                            )
                        min_dir_component_abs = float(
                            special_solver_cfg.get("min_dir_component_abs", 0.2)
                        )
                        if min_dir_component_abs <= 0.0 or min_dir_component_abs > 1.0:
                            raise RuntimeError(
                                f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                                "special_solver.min_dir_component_abs must be in (0, 1]"
                            )
                        x_beam: Optional[str] = None
                        y_beam: Optional[str] = None
                        if "x" in solve_axes:
                            x_beam = str(special_solver_cfg.get("x_beam", ""))
                            if x_beam not in PROJECTED_ALLOWED_BEAMS:
                                raise RuntimeError(
                                    f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                                    f"special_solver.x_beam must be one of {SENSOR_ORDER}"
                                )
                            max_x_correction_m = float(
                                special_solver_cfg.get("max_x_correction_m", 0.15)
                            )
                            if max_x_correction_m < 0.0:
                                raise RuntimeError(
                                    f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                                    "special_solver.max_x_correction_m must be >= 0"
                                )
                        if "y" in solve_axes:
                            y_beam = str(special_solver_cfg.get("y_beam", ""))
                            if y_beam not in PROJECTED_ALLOWED_BEAMS:
                                raise RuntimeError(
                                    f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                                    f"special_solver.y_beam must be one of {SENSOR_ORDER}"
                                )
                            max_y_correction_m = float(
                                special_solver_cfg.get("max_y_correction_m", 0.15)
                            )
                            if max_y_correction_m < 0.0:
                                raise RuntimeError(
                                    f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                                    "special_solver.max_y_correction_m must be >= 0"
                                )
                        if x_beam is not None and y_beam is not None and x_beam == y_beam:
                            raise RuntimeError(
                                f"scene_profiles.{scene_name}.wall_selector.regions[{index}] "
                                "special_solver.x_beam and special_solver.y_beam must be different when solving both axes"
                            )

    def _projected_xy_solve_axes(
        self, special_solver_cfg: Dict[str, Any]
    ) -> List[str]:
        raw_axes = special_solver_cfg.get("solve_axes", [])
        if not isinstance(raw_axes, list):
            return []
        return [str(axis).strip().lower() for axis in raw_axes]

    def _region_wall_pair_names(self, region: Dict[str, Any]) -> List[str]:
        if "active_wall_pair" in region:
            pair_name = region.get("active_wall_pair")
            if pair_name is None or isinstance(pair_name, list):
                return []
            return [str(pair_name)]
        if "active_wall_pairs" in region:
            pair_names = region.get("active_wall_pairs")
            if not isinstance(pair_names, list) or len(pair_names) != 2:
                return []
            return [str(pair_name) for pair_name in pair_names]
        return []

    def _special_solver_type(self, region: Optional[Dict[str, Any]]) -> Optional[str]:
        if not isinstance(region, dict):
            return None
        special_solver_cfg = region.get("special_solver")
        if not isinstance(special_solver_cfg, dict):
            return None
        solver_type = special_solver_cfg.get("type")
        if solver_type is None:
            return None
        return str(solver_type)

    def _region_xy_yaw_source(self, region: Optional[Dict[str, Any]]) -> str:
        if not isinstance(region, dict):
            return "side_laser"
        raw_source = region.get("xy_yaw_source")
        if raw_source is None and self._special_solver_type(region) == "projected_xy_with_lidar_yaw":
            return "lidar"
        return str(region.get("xy_yaw_source", "side_laser")).strip().lower()

    def _resolve_xy_projection_yaw(
        self,
        *,
        coarse_yaw_deg: float,
        wall_pair: WallPair,
        side_laser_yaw_corner_deg: float,
        region_cfg: Optional[Dict[str, Any]],
    ) -> Tuple[str, float, float]:
        lidar_yaw_corner_deg = wrap_deg(coarse_yaw_deg - wall_pair.corner_yaw_deg)
        xy_yaw_source = self._region_xy_yaw_source(region_cfg)
        xy_projection_yaw_corner_deg = (
            lidar_yaw_corner_deg
            if xy_yaw_source == "lidar"
            else side_laser_yaw_corner_deg
        )
        return xy_yaw_source, lidar_yaw_corner_deg, xy_projection_yaw_corner_deg

    def _is_finite_pose(self, coarse: CoarsePose) -> bool:
        return (
            math.isfinite(coarse.x)
            and math.isfinite(coarse.y)
            and math.isfinite(coarse.z)
            and math.isfinite(coarse.roll_rad)
            and math.isfinite(coarse.pitch_rad)
            and math.isfinite(coarse.yaw_deg)
        )

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
        if best_region is None:
            return None, {
                "evaluated": True,
                "matched": False,
                "matched_region_name": None,
                "matched_wall_pair_name": None,
                "matched_wall_pair_names": None,
                "candidate_count": len(candidate_debug),
                "candidates": candidate_debug,
            }

        pair_names = self._region_wall_pair_names(best_region)
        if not pair_names:
            return None, {
                "evaluated": True,
                "matched": False,
                "matched_region_name": str(best_region.get("name", "")),
                "matched_wall_pair_name": None,
                "matched_wall_pair_names": None,
                "candidate_count": len(candidate_debug),
                "candidates": candidate_debug,
            }
        wall_pairs = [self.wall_pairs[pair_name] for pair_name in pair_names]
        matched_region_name = str(best_region.get("name", ""))
        for candidate in candidate_debug:
            if candidate.get("name") == matched_region_name:
                candidate["matched"] = True

        return RegionMatch(
            name=matched_region_name,
            wall_pairs=wall_pairs,
            region_config=best_region,
        ), {
            "evaluated": True,
            "matched": bool(wall_pairs),
            "matched_region_name": matched_region_name,
            "matched_wall_pair_name": pair_names[0] if len(pair_names) == 1 else None,
            "matched_wall_pair_names": pair_names,
            "matched_special_solver_type": self._special_solver_type(best_region),
            "matched_xy_yaw_source": self._region_xy_yaw_source(best_region),
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
            "active_wall_pair_names": self._region_wall_pair_names(region),
            "xy_yaw_source": self._region_xy_yaw_source(region),
            "position_score_m": self._debug_float(position_score),
            "yaw_error_deg": self._debug_float(yaw_error_deg),
            "expected_yaw_deg": (
                float(region["yaw_deg"]) if "yaw_deg" in region else None
            ),
            "special_solver_type": self._special_solver_type(region),
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
        publish_x: Optional[float] = None,
        publish_y: Optional[float] = None,
        publish_z: Optional[float] = None,
        publish_yaw_deg: Optional[float] = None,
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
            publish_x=publish_x,
            publish_y=publish_y,
            publish_z=publish_z,
            publish_yaw_deg=publish_yaw_deg,
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
            publish_x=None,
            publish_y=None,
            publish_z=None,
            publish_yaw_deg=None,
        )

    def _full_pose_publish_fields(
        self,
        coarse: CoarsePose,
        *,
        x: float,
        y: float,
        yaw_deg: Optional[float],
    ) -> Dict[str, Optional[float]]:
        return {
            "publish_x": x,
            "publish_y": y,
            "publish_z": coarse.z,
            "publish_yaw_deg": yaw_deg,
        }

    def _single_axis_publish_fields(
        self,
        *,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ) -> Dict[str, Optional[float]]:
        return {
            "publish_x": x,
            "publish_y": y,
            "publish_z": None,
            "publish_yaw_deg": None,
        }

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

    def _solve_dual_wall_pair_region(
        self,
        coarse: CoarsePose,
        range_frame: RangeFrame,
        wall_pairs: List[WallPair],
        solver_cfg: Dict[str, Any],
        region_cfg: Optional[Dict[str, Any]],
        prior_age_ms: float,
        region_name: Optional[str] = None,
        usable_sensor_count: int = 0,
        debug: Optional[Dict[str, Any]] = None,
    ) -> SolveResult:
        candidate_results = [
            self._solve_closed_form(
                coarse=coarse,
                range_frame=range_frame,
                wall_pair=wall_pair,
                solver_cfg=solver_cfg,
                region_cfg=region_cfg,
                prior_age_ms=prior_age_ms,
                region_name=region_name,
                usable_sensor_count=usable_sensor_count,
                debug=debug,
            )
            for wall_pair in wall_pairs
        ]
        credible_indices = [
            index
            for index, candidate_result in enumerate(candidate_results)
            if self._is_dual_wall_pair_candidate_credible(candidate_result, solver_cfg)
        ]
        selection_indices = (
            credible_indices if credible_indices else list(range(len(candidate_results)))
        )
        selected_index = self._select_best_solver_candidate_index(
            candidate_results,
            candidate_indices=selection_indices,
        )
        preferred_result = candidate_results[selected_index]
        fallback_used = not credible_indices
        selected_result = (
            self._build_dual_wall_pair_front_x_fallback_result(
                coarse=coarse,
                range_frame=range_frame,
                wall_pairs=wall_pairs,
                solver_cfg=solver_cfg,
                prior_age_ms=prior_age_ms,
                region_name=region_name,
                usable_sensor_count=usable_sensor_count,
                debug=debug,
            )
            if fallback_used
            else preferred_result
        )
        selected_debug = dict(selected_result.debug or {})
        region_debug = dict(selected_debug.get("region_debug", {}))
        region_debug.update(
            {
                "multi_wall_pair_mode": True,
                "matched_wall_pair_name": None,
                "matched_wall_pair_names": [
                    wall_pair.name for wall_pair in wall_pairs
                ],
                "selected_wall_pair_name": selected_result.wall_pair_name,
                "preferred_solver_candidate_wall_pair_name": preferred_result.wall_pair_name,
                "selected_solver_candidate_index": selected_index,
                "solver_candidate_count": len(candidate_results),
                "credible_solver_candidate_indices": credible_indices,
                "credible_solver_candidate_count": len(credible_indices),
                "dual_wall_pair_front_x_fallback_used": fallback_used,
                "solver_candidates": [
                    self._build_solver_candidate_debug(
                        result=candidate_result,
                        selected=index == selected_index and not fallback_used,
                        preferred=index == selected_index,
                        credible=index in credible_indices,
                    )
                    for index, candidate_result in enumerate(candidate_results)
                ],
                "selection_strategy": (
                    "prefer_credible_candidates_then_lower_residual_then_more_hits_then_closer_to_lidar"
                ),
            }
        )
        selected_debug["region_debug"] = region_debug
        return replace(selected_result, debug=selected_debug)

    def _select_best_solver_candidate_index(
        self,
        candidate_results: List[SolveResult],
        candidate_indices: Optional[List[int]] = None,
    ) -> int:
        if not candidate_results:
            raise RuntimeError("dual wall-pair region produced no solver candidates")
        if candidate_indices is None:
            candidate_indices = list(range(len(candidate_results)))
        if not candidate_indices:
            raise RuntimeError("dual wall-pair region candidate index set is empty")
        best_index = candidate_indices[0]
        best_key = self._solver_candidate_sort_key(candidate_results[best_index])
        for index in candidate_indices[1:]:
            candidate_result = candidate_results[index]
            candidate_key = self._solver_candidate_sort_key(candidate_result)
            if candidate_key < best_key:
                best_index = index
                best_key = candidate_key
        return best_index

    def _solver_candidate_sort_key(
        self, result: SolveResult
    ) -> Tuple[int, float, int, float, float, int, float, str]:
        if result.state == STATE_REFINED:
            state_rank = 0
        elif result.state == STATE_COARSE_ONLY:
            state_rank = 1
        else:
            state_rank = 2
        delta_xy_norm_m, delta_yaw_deg = self._result_correction_metrics(result)
        residual_rank = (
            float(result.residual_m)
            if result.residual_m is not None
            else float("inf")
        )
        return (
            state_rank,
            residual_rank,
            -int(result.target_hit_count),
            delta_xy_norm_m,
            delta_yaw_deg,
            -int(result.selected_valid_beam_count),
            -float(result.score),
            str(result.reason),
        )

    def _result_correction_metrics(self, result: SolveResult) -> Tuple[float, float]:
        correction_debug = (
            (result.debug or {})
            .get("solver_debug", {})
            .get("correction_debug", {})
        )
        delta_xy_norm_m = correction_debug.get("delta_xy_norm_m")
        delta_yaw_deg = correction_debug.get("delta_yaw_deg")
        if delta_xy_norm_m is None or not math.isfinite(float(delta_xy_norm_m)):
            delta_xy_norm_m = float("inf")
        else:
            delta_xy_norm_m = float(delta_xy_norm_m)
        if delta_yaw_deg is None or not math.isfinite(float(delta_yaw_deg)):
            delta_yaw_deg = float("inf")
        else:
            delta_yaw_deg = abs(float(delta_yaw_deg))
        return delta_xy_norm_m, delta_yaw_deg

    def _build_solver_candidate_debug(
        self,
        result: SolveResult,
        *,
        selected: bool,
        preferred: bool,
        credible: bool,
    ) -> Dict[str, Any]:
        delta_xy_norm_m, delta_yaw_deg = self._result_correction_metrics(result)
        candidate_debug = {
            "selected": selected,
            "preferred": preferred,
            "credible": credible,
            "state": result.state,
            "reason": result.reason,
            "localized": result.localized,
            "wall_pair_name": result.wall_pair_name,
            "beam_mode": result.beam_mode,
            "score": self._debug_float(result.score),
            "residual_m": self._debug_float(result.residual_m),
            "target_hit_count": int(result.target_hit_count),
            "selected_beam_count": int(result.selected_beam_count),
            "selected_valid_beam_count": int(result.selected_valid_beam_count),
            "delta_xy_norm_m": self._debug_float(delta_xy_norm_m),
            "delta_yaw_deg": self._debug_float(delta_yaw_deg),
        }
        solver_debug = (result.debug or {}).get("solver_debug", {})
        candidate_pose = solver_debug.get("candidate_pose")
        if isinstance(candidate_pose, dict):
            candidate_debug["candidate_pose"] = {
                "x": self._debug_float(candidate_pose.get("x")),
                "y": self._debug_float(candidate_pose.get("y")),
                "yaw_deg": self._debug_float(candidate_pose.get("yaw_deg")),
            }
        return candidate_debug

    def _is_dual_wall_pair_candidate_credible(
        self,
        result: SolveResult,
        solver_cfg: Dict[str, Any],
    ) -> bool:
        if result.state != STATE_REFINED:
            return False
        if (
            result.x is None
            or result.y is None
            or result.yaw_deg is None
            or not math.isfinite(float(result.x))
            or not math.isfinite(float(result.y))
            or not math.isfinite(float(result.yaw_deg))
        ):
            return False

        residual_thresh_m = float(solver_cfg.get("residual_thresh_m", 0.03))
        min_valid_corner_beams = int(solver_cfg.get("min_valid_corner_beams", 3))
        max_correction_xy_m = float(solver_cfg.get("max_correction_xy_m", 0.15))
        max_correction_yaw_deg = float(solver_cfg.get("max_correction_yaw_deg", 10.0))

        if result.residual_m is None or not math.isfinite(float(result.residual_m)):
            return False
        if float(result.residual_m) > residual_thresh_m:
            return False
        if int(result.target_hit_count) < min_valid_corner_beams:
            return False

        delta_xy_norm_m, delta_yaw_deg = self._result_correction_metrics(result)
        if delta_xy_norm_m > max_correction_xy_m:
            return False
        if delta_yaw_deg > max_correction_yaw_deg:
            return False
        return True

    def _build_dual_wall_pair_front_x_fallback_result(
        self,
        coarse: CoarsePose,
        range_frame: RangeFrame,
        wall_pairs: List[WallPair],
        solver_cfg: Dict[str, Any],
        prior_age_ms: float,
        region_name: Optional[str] = None,
        usable_sensor_count: int = 0,
        debug: Optional[Dict[str, Any]] = None,
    ) -> SolveResult:
        front_beam = "front_center"
        beam_mode = "front_x_lidar_yaw_fallback"
        selected_beams = [front_beam]
        selected_beam_count = len(selected_beams)
        selected_valid_beam_count = (
            1 if range_frame.valid.get(front_beam, False) else 0
        )
        fallback_debug = dict(debug or {})
        solver_debug = dict(fallback_debug.get("solver_debug", {}))
        solver_debug.update(
            {
                "beam_mode": beam_mode,
                "x_beam": front_beam,
                "fallback_mode": "front_x_with_lidar_yaw",
                "fallback_trigger": "all_dual_wall_pair_candidates_rejected",
                "front_beam_valid": bool(range_frame.valid.get(front_beam, False)),
            }
        )
        fallback_debug["solver_debug"] = solver_debug

        if not range_frame.valid.get(front_beam, False):
            solver_debug["fallback_failure_reason"] = "FRONT_BEAM_INVALID"
            fallback_debug["solver_debug"] = solver_debug
            return self._make_coarse_result(
                coarse,
                reason="DUAL_WALL_PAIR_FRONT_X_FALLBACK_UNAVAILABLE",
                valid_beam_count=selected_valid_beam_count,
                prior_age_ms=prior_age_ms,
                usable_sensor_count=usable_sensor_count,
                selected_beam_count=selected_beam_count,
                selected_valid_beam_count=selected_valid_beam_count,
                debug=fallback_debug,
                region_name=region_name,
                beam_mode=beam_mode,
                selected_beams=selected_beams,
            )

        front_wall, failure_reason = self._resolve_dual_wall_pair_front_x_fallback_wall(
            wall_pairs
        )
        if front_wall is None:
            solver_debug["fallback_failure_reason"] = failure_reason
            fallback_debug["solver_debug"] = solver_debug
            return self._make_coarse_result(
                coarse,
                reason="DUAL_WALL_PAIR_FRONT_X_FALLBACK_UNAVAILABLE",
                valid_beam_count=selected_valid_beam_count,
                prior_age_ms=prior_age_ms,
                usable_sensor_count=usable_sensor_count,
                selected_beam_count=selected_beam_count,
                selected_valid_beam_count=selected_valid_beam_count,
                debug=fallback_debug,
                region_name=region_name,
                beam_mode=beam_mode,
                selected_beams=selected_beams,
            )

        front_range = range_frame.ranges.get(front_beam)
        if front_range is None or not math.isfinite(float(front_range)):
            solver_debug["fallback_failure_reason"] = "FRONT_RANGE_NON_FINITE"
            fallback_debug["solver_debug"] = solver_debug
            return self._make_coarse_result(
                coarse,
                reason="DUAL_WALL_PAIR_FRONT_X_FALLBACK_UNAVAILABLE",
                valid_beam_count=selected_valid_beam_count,
                prior_age_ms=prior_age_ms,
                usable_sensor_count=usable_sensor_count,
                selected_beam_count=selected_beam_count,
                selected_valid_beam_count=selected_valid_beam_count,
                debug=fallback_debug,
                region_name=region_name,
                beam_mode=beam_mode,
                selected_beams=selected_beams,
            )

        mount = self.sensor_mounts[front_beam]
        origin_dx, origin_dy = rotate_2d(mount.pos_x, mount.pos_y, coarse.yaw_deg)
        dir_dx, dir_dy = rotate_2d(mount.dir_x, mount.dir_y, coarse.yaw_deg)
        if front_wall.orientation == "vertical":
            if abs(dir_dx) <= 1e-6:
                solver_debug["fallback_failure_reason"] = (
                    "FRONT_BEAM_PARALLEL_TO_FRONT_WALL_NORMAL"
                )
                fallback_debug["solver_debug"] = solver_debug
                return self._make_coarse_result(
                    coarse,
                    reason="DUAL_WALL_PAIR_FRONT_X_FALLBACK_UNAVAILABLE",
                    valid_beam_count=selected_valid_beam_count,
                    prior_age_ms=prior_age_ms,
                    usable_sensor_count=usable_sensor_count,
                    selected_beam_count=selected_beam_count,
                    selected_valid_beam_count=selected_valid_beam_count,
                    debug=fallback_debug,
                    region_name=region_name,
                    beam_mode=beam_mode,
                    selected_beams=selected_beams,
                )
            x_map = float(front_wall.const_value) - origin_dx - float(front_range) * dir_dx
            y_map = coarse.y
            delta_x = x_map - coarse.x
            delta_y = 0.0
            solved_axis = "x"
        elif front_wall.orientation == "horizontal":
            if abs(dir_dy) <= 1e-6:
                solver_debug["fallback_failure_reason"] = (
                    "FRONT_BEAM_PARALLEL_TO_FRONT_WALL_NORMAL"
                )
                fallback_debug["solver_debug"] = solver_debug
                return self._make_coarse_result(
                    coarse,
                    reason="DUAL_WALL_PAIR_FRONT_X_FALLBACK_UNAVAILABLE",
                    valid_beam_count=selected_valid_beam_count,
                    prior_age_ms=prior_age_ms,
                    usable_sensor_count=usable_sensor_count,
                    selected_beam_count=selected_beam_count,
                    selected_valid_beam_count=selected_valid_beam_count,
                    debug=fallback_debug,
                    region_name=region_name,
                    beam_mode=beam_mode,
                    selected_beams=selected_beams,
                )
            x_map = coarse.x
            y_map = float(front_wall.const_value) - origin_dy - float(front_range) * dir_dy
            delta_x = 0.0
            delta_y = y_map - coarse.y
            solved_axis = "y"
        else:
            solver_debug["fallback_failure_reason"] = "FRONT_WALL_IS_NOT_AXIS_ALIGNED"
            fallback_debug["solver_debug"] = solver_debug
            return self._make_coarse_result(
                coarse,
                reason="DUAL_WALL_PAIR_FRONT_X_FALLBACK_UNAVAILABLE",
                valid_beam_count=selected_valid_beam_count,
                prior_age_ms=prior_age_ms,
                usable_sensor_count=usable_sensor_count,
                selected_beam_count=selected_beam_count,
                selected_valid_beam_count=selected_valid_beam_count,
                debug=fallback_debug,
                region_name=region_name,
                beam_mode=beam_mode,
                selected_beams=selected_beams,
            )
        max_correction_xy_m = float(solver_cfg.get("max_correction_xy_m", 0.15))
        max_correction_yaw_deg = float(solver_cfg.get("max_correction_yaw_deg", 10.0))
        fallback_debug = self._with_solver_debug(
            fallback_debug,
            candidate_pose={
                "x": x_map,
                "y": y_map,
                "yaw_deg": coarse.yaw_deg,
            },
            correction_debug={
                "delta_x_m": delta_x,
                "delta_y_m": delta_y,
                "delta_xy_norm_m": math.hypot(delta_x, delta_y),
                "delta_yaw_deg": 0.0,
                "max_correction_xy_m": max_correction_xy_m,
                "max_correction_yaw_deg": max_correction_yaw_deg,
            },
        )
        solver_debug = dict(fallback_debug.get("solver_debug", {}))
        solver_debug.update(
            {
                "front_wall_name": front_wall.name,
                "front_wall_orientation": front_wall.orientation,
                "front_wall_const_value_m": self._debug_float(front_wall.const_value),
                "front_beam_range_m": self._debug_float(float(front_range)),
                "fallback_solved_axis": solved_axis,
            }
        )
        if front_wall.orientation == "vertical":
            solver_debug["front_wall_const_x_m"] = self._debug_float(
                front_wall.const_value
            )
        else:
            solver_debug["front_wall_const_y_m"] = self._debug_float(
                front_wall.const_value
            )
        fallback_debug["solver_debug"] = solver_debug
        return self._make_result(
            state=STATE_REFINED,
            reason="OK",
            pose_source="front_laser_x_with_lidar_yaw",
            localized=True,
            x=x_map,
            y=y_map,
            yaw_deg=coarse.yaw_deg,
            valid_beam_count=selected_valid_beam_count,
            prior_age_ms=prior_age_ms,
            usable_sensor_count=usable_sensor_count,
            selected_beam_count=selected_beam_count,
            selected_valid_beam_count=selected_valid_beam_count,
            target_hit_count=selected_valid_beam_count,
            debug=fallback_debug,
            region_name=region_name,
            beam_mode=beam_mode,
            selected_beams=selected_beams,
            **self._single_axis_publish_fields(
                x=x_map if solved_axis == "x" else None,
                y=y_map if solved_axis == "y" else None,
            ),
        )

    def _resolve_dual_wall_pair_front_x_fallback_wall(
        self,
        wall_pairs: List[WallPair],
    ) -> Tuple[Optional[WallSegment], Optional[str]]:
        if not wall_pairs:
            return None, "NO_DUAL_WALL_PAIRS"

        x_walls: List[WallSegment] = []
        for wall_pair in wall_pairs:
            if wall_pair.x_wall_role != "front":
                return None, "X_WALL_ROLE_IS_NOT_FRONT"
            x_wall = self.walls.get(wall_pair.x_wall_name)
            if x_wall is None:
                return None, "X_WALL_MISSING"
            if x_wall.orientation not in {"vertical", "horizontal"}:
                return None, "X_WALL_IS_NOT_AXIS_ALIGNED"
            x_walls.append(x_wall)

        ref_wall = x_walls[0]
        for x_wall in x_walls[1:]:
            if x_wall.orientation != ref_wall.orientation:
                return None, "X_WALLS_NOT_ALIGNED"
            if not math.isclose(
                float(x_wall.const_value),
                float(ref_wall.const_value),
                abs_tol=1e-6,
            ):
                return None, "X_WALLS_NOT_COLOCATED"
        return ref_wall, None

    def _solve_closed_form(
        self,
        coarse: CoarsePose,
        range_frame: RangeFrame,
        wall_pair: WallPair,
        solver_cfg: Dict[str, Any],
        region_cfg: Optional[Dict[str, Any]],
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
        (
            xy_yaw_source,
            lidar_yaw_corner_deg,
            xy_projection_yaw_corner_deg,
        ) = self._resolve_xy_projection_yaw(
            coarse_yaw_deg=coarse.yaw_deg,
            wall_pair=wall_pair,
            side_laser_yaw_corner_deg=theta_side_deg,
            region_cfg=region_cfg,
        )

        max_theta_abs_deg = float(solver_cfg.get("max_theta_abs_deg", 45.0))
        debug = self._with_solver_debug(
            debug,
            beam_selection=beam_selection,
            theta_side_deg=theta_side_deg,
        )
        debug = self._with_xy_yaw_debug(
            debug,
            xy_yaw_source=xy_yaw_source,
            side_laser_yaw_in_corner_deg=theta_side_deg,
            lidar_yaw_in_corner_deg=lidar_yaw_corner_deg,
            xy_projection_yaw_in_corner_deg=xy_projection_yaw_corner_deg,
        )
        if abs(xy_projection_yaw_corner_deg) > max_theta_abs_deg:
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

        c = math.cos(math.radians(xy_projection_yaw_corner_deg))
        sx = -1.0 if beam_selection.x_beam_role == "front" else 1.0
        sy = 1.0 if beam_selection.side_beam_role == "right" else -1.0
        d_x = beam_selection.x_offset_m
        d_y = beam_selection.side_offset_m

        x_corner = sx * (r_x + d_x) * c
        y_corner = sy * (d_y + 0.5 * (r_sf + r_sr)) * c
        yaw_corner_deg = xy_projection_yaw_corner_deg

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
            **self._full_pose_publish_fields(
                coarse,
                x=x_map,
                y=y_map,
                yaw_deg=yaw_map_deg if xy_yaw_source != "lidar" else None,
            ),
        )

    def _solve_compensated_front_side_region(
        self,
        coarse: CoarsePose,
        range_frame: RangeFrame,
        wall_pair: WallPair,
        solver_cfg: Dict[str, Any],
        special_solver_cfg: Dict[str, Any],
        region_cfg: Optional[Dict[str, Any]],
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
        compensation_x0_m = float(special_solver_cfg.get("compensation_x0_m", 0.125))
        max_iterations = int(special_solver_cfg.get("max_iterations", 8))
        theta_tolerance_deg = float(
            special_solver_cfg.get("theta_tolerance_deg", 0.01)
        )
        lidar_yaw_corner_deg = beam_selection.yaw_in_corner_deg
        xy_yaw_source = self._region_xy_yaw_source(region_cfg)

        debug = self._with_solver_debug(debug, beam_selection=beam_selection)
        debug = self._with_special_solver_debug(
            debug,
            solver_type="compensated_front_side",
            compensation_x0_m=compensation_x0_m,
            max_iterations=max_iterations,
            theta_tolerance_deg=theta_tolerance_deg,
            theta_seed_deg=beam_selection.yaw_in_corner_deg,
        )
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
        if beam_selection.x_beam_role != "front":
            debug = self._with_special_solver_debug(
                debug,
                failure_reason="SPECIAL_SOLVER_REQUIRES_FRONT_X_BEAM",
            )
            return self._make_coarse_result(
                coarse,
                reason="SPECIAL_SOLVER_CONFIGURATION_ERROR",
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

        front_range = range_frame.ranges[beam_selection.x_beam]
        side_front_range = range_frame.ranges[beam_selection.side_front_beam]
        side_rear_range = range_frame.ranges[beam_selection.side_rear_beam]
        debug = self._with_special_solver_debug(
            debug,
            front_range_m=front_range,
            side_front_range_m=side_front_range,
            side_rear_range_m=side_rear_range,
            pair_spacing_m=beam_selection.pair_spacing_m,
            side_beam_role=beam_selection.side_beam_role,
        )
        theta_side_deg: Optional[float]
        corrected_side_front_range: Optional[float]
        theta_iterations: int
        theta_converged: Optional[bool]
        theta_failure_reason: Optional[str]
        if xy_yaw_source == "lidar":
            theta_side_deg = None
            theta_iterations = 0
            corrected_side_front_range, theta_failure_reason = (
                self._compensate_side_front_range_for_yaw(
                    side_front_range=side_front_range,
                    compensation_x0_m=compensation_x0_m,
                    yaw_corner_deg=lidar_yaw_corner_deg,
                )
            )
            theta_converged = theta_failure_reason is None
        else:
            (
                theta_side_deg,
                corrected_side_front_range,
                theta_iterations,
                theta_converged,
                theta_failure_reason,
            ) = self._iterate_compensated_theta(
                side_beam_role=beam_selection.side_beam_role,
                side_front_range=side_front_range,
                side_rear_range=side_rear_range,
                pair_spacing_m=beam_selection.pair_spacing_m,
                compensation_x0_m=compensation_x0_m,
                theta_seed_deg=beam_selection.yaw_in_corner_deg,
                max_iterations=max_iterations,
                theta_tolerance_deg=theta_tolerance_deg,
            )
        xy_projection_yaw_corner_deg = (
            lidar_yaw_corner_deg if xy_yaw_source == "lidar" else theta_side_deg
        )
        debug = self._with_special_solver_debug(
            debug,
            theta_iterations=theta_iterations,
            theta_converged=theta_converged,
            corrected_side_front_range_m=corrected_side_front_range,
            failure_reason=theta_failure_reason,
        )
        debug = self._with_xy_yaw_debug(
            debug,
            xy_yaw_source=xy_yaw_source,
            side_laser_yaw_in_corner_deg=theta_side_deg,
            lidar_yaw_in_corner_deg=lidar_yaw_corner_deg,
            xy_projection_yaw_in_corner_deg=xy_projection_yaw_corner_deg,
        )
        if xy_projection_yaw_corner_deg is None or corrected_side_front_range is None:
            return self._make_coarse_result(
                coarse,
                reason="SPECIAL_SOLVER_THETA_FAILED",
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

        max_theta_abs_deg = float(solver_cfg.get("max_theta_abs_deg", 45.0))
        debug = self._with_solver_debug(
            debug,
            beam_selection=beam_selection,
            theta_side_deg=theta_side_deg,
        )
        if abs(xy_projection_yaw_corner_deg) > max_theta_abs_deg:
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

        c = math.cos(math.radians(xy_projection_yaw_corner_deg))
        if abs(c) <= 1e-6:
            debug = self._with_special_solver_debug(
                debug,
                failure_reason="THETA_COMPENSATION_SINGULAR",
            )
            return self._make_coarse_result(
                coarse,
                reason="SPECIAL_SOLVER_THETA_FAILED",
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

        sx = -1.0
        sy = 1.0 if beam_selection.side_beam_role == "right" else -1.0
        d_x = beam_selection.x_offset_m
        d_y = beam_selection.side_offset_m

        x_corner = sx * (front_range + d_x) * c
        y_corner = sy * (d_y + 0.5 * (corrected_side_front_range + side_rear_range)) * c
        yaw_corner_deg = xy_projection_yaw_corner_deg

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
            measured_range_overrides={
                beam_selection.side_front_beam: corrected_side_front_range
            },
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
        debug = self._with_special_solver_debug(
            debug,
            corrected_side_front_range_m=corrected_side_front_range,
        )

        return self._make_result(
            state=STATE_REFINED,
            reason="OK",
            pose_source="compensated_front_side_corner_solver",
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
            **self._full_pose_publish_fields(
                coarse,
                x=x_map,
                y=y_map,
                yaw_deg=yaw_map_deg if xy_yaw_source != "lidar" else None,
            ),
        )

    def _solve_projected_xy_with_lidar_yaw(
        self,
        coarse: CoarsePose,
        range_frame: RangeFrame,
        wall_pair: WallPair,
        solver_cfg: Dict[str, Any],
        special_solver_cfg: Dict[str, Any],
        prior_age_ms: float,
        region_name: Optional[str] = None,
        usable_sensor_count: int = 0,
        debug: Optional[Dict[str, Any]] = None,
    ) -> SolveResult:
        solve_axes = self._projected_xy_solve_axes(special_solver_cfg)
        x_beam = (
            str(special_solver_cfg.get("x_beam", "front_center"))
            if "x" in solve_axes
            else None
        )
        y_beam = (
            str(special_solver_cfg.get("y_beam", "left_front"))
            if "y" in solve_axes
            else None
        )
        selected_beams = [
            beam_name for beam_name in [x_beam, y_beam] if beam_name is not None
        ]
        selected_beam_count = len(selected_beams)
        selected_valid_beam_count = sum(
            1 for beam_name in selected_beams if range_frame.valid.get(beam_name, False)
        )
        min_dir_component_abs = float(
            special_solver_cfg.get("min_dir_component_abs", 0.2)
        )
        max_x_correction_m = float(
            special_solver_cfg.get(
                "max_x_correction_m", solver_cfg.get("max_correction_xy_m", 0.15)
            )
        )
        max_y_correction_m = float(
            special_solver_cfg.get(
                "max_y_correction_m", solver_cfg.get("max_correction_xy_m", 0.15)
            )
        )
        max_correction_yaw_deg = float(solver_cfg.get("max_correction_yaw_deg", 10.0))
        coarse_corner_x, coarse_corner_y, yaw_corner_deg = self._world_pose_to_corner_local(
            pose_x=coarse.x,
            pose_y=coarse.y,
            pose_yaw_deg=coarse.yaw_deg,
            wall_pair=wall_pair,
        )

        debug = self._with_special_solver_debug(
            debug,
            solver_type="projected_xy_with_lidar_yaw",
            min_dir_component_abs=min_dir_component_abs,
            solve_x=1.0 if "x" in solve_axes else 0.0,
            solve_y=1.0 if "y" in solve_axes else 0.0,
            x_wall_name=wall_pair.x_wall_name,
            side_wall_name=wall_pair.side_wall_name,
            x_wall_role=wall_pair.x_wall_role,
            side_wall_role=wall_pair.side_wall_role,
            yaw_in_corner_deg=yaw_corner_deg,
        )
        debug = self._with_xy_yaw_debug(
            debug,
            xy_yaw_source="lidar",
            side_laser_yaw_in_corner_deg=None,
            lidar_yaw_in_corner_deg=yaw_corner_deg,
            xy_projection_yaw_in_corner_deg=yaw_corner_deg,
        )

        x_corner = coarse_corner_x
        y_corner = coarse_corner_y
        solve_debug: Dict[str, Any] = {}

        if "x" in solve_axes:
            if x_beam is None or not range_frame.valid.get(x_beam, False):
                debug = self._with_special_solver_debug(
                    debug,
                    failure_reason="PROJECTED_X_BEAM_INVALID",
                )
                return self._make_coarse_result(
                    coarse,
                    reason="PROJECTED_AXIS_UNAVAILABLE",
                    valid_beam_count=selected_valid_beam_count,
                    prior_age_ms=prior_age_ms,
                    usable_sensor_count=usable_sensor_count,
                    selected_beam_count=selected_beam_count,
                    selected_valid_beam_count=selected_valid_beam_count,
                    debug=debug,
                    wall_pair_name=wall_pair.name,
                    region_name=region_name,
                    beam_mode="projected_xy_with_lidar_yaw",
                    selected_beams=selected_beams,
                    yaw_in_corner_deg=yaw_corner_deg,
                )
            x_value, x_debug = self._solve_projected_corner_axis_from_beam(
                beam_name=x_beam,
                target_wall_name=wall_pair.x_wall_name,
                target_wall_role=wall_pair.x_wall_role,
                pose_yaw_deg=yaw_corner_deg,
                measured_range=range_frame.ranges[x_beam],
                axis="x",
                min_dir_component_abs=min_dir_component_abs,
            )
            solve_debug["x_solver"] = x_debug
            if x_value is None:
                debug = self._merge_special_solver_debug_map(debug, solve_debug)
                debug = self._with_special_solver_debug(
                    debug,
                    failure_reason="PROJECTED_X_GEOMETRY_UNAVAILABLE",
                )
                return self._make_coarse_result(
                    coarse,
                    reason="PROJECTED_AXIS_UNAVAILABLE",
                    valid_beam_count=selected_valid_beam_count,
                    prior_age_ms=prior_age_ms,
                    usable_sensor_count=usable_sensor_count,
                    selected_beam_count=selected_beam_count,
                    selected_valid_beam_count=selected_valid_beam_count,
                    debug=debug,
                    wall_pair_name=wall_pair.name,
                    region_name=region_name,
                    beam_mode="projected_xy_with_lidar_yaw",
                    selected_beams=selected_beams,
                    yaw_in_corner_deg=yaw_corner_deg,
                )
            x_corner = x_value

        if "y" in solve_axes:
            if y_beam is None or not range_frame.valid.get(y_beam, False):
                debug = self._merge_special_solver_debug_map(debug, solve_debug)
                debug = self._with_special_solver_debug(
                    debug,
                    failure_reason="PROJECTED_Y_BEAM_INVALID",
                )
                return self._make_coarse_result(
                    coarse,
                    reason="PROJECTED_AXIS_UNAVAILABLE",
                    valid_beam_count=selected_valid_beam_count,
                    prior_age_ms=prior_age_ms,
                    usable_sensor_count=usable_sensor_count,
                    selected_beam_count=selected_beam_count,
                    selected_valid_beam_count=selected_valid_beam_count,
                    debug=debug,
                    wall_pair_name=wall_pair.name,
                    region_name=region_name,
                    beam_mode="projected_xy_with_lidar_yaw",
                    selected_beams=selected_beams,
                    yaw_in_corner_deg=yaw_corner_deg,
                )
            y_value, y_debug = self._solve_projected_corner_axis_from_beam(
                beam_name=y_beam,
                target_wall_name=wall_pair.side_wall_name,
                target_wall_role=wall_pair.side_wall_role,
                pose_yaw_deg=yaw_corner_deg,
                measured_range=range_frame.ranges[y_beam],
                axis="y",
                min_dir_component_abs=min_dir_component_abs,
            )
            solve_debug["y_solver"] = y_debug
            if y_value is None:
                debug = self._merge_special_solver_debug_map(debug, solve_debug)
                debug = self._with_special_solver_debug(
                    debug,
                    failure_reason="PROJECTED_Y_GEOMETRY_UNAVAILABLE",
                )
                return self._make_coarse_result(
                    coarse,
                    reason="PROJECTED_AXIS_UNAVAILABLE",
                    valid_beam_count=selected_valid_beam_count,
                    prior_age_ms=prior_age_ms,
                    usable_sensor_count=usable_sensor_count,
                    selected_beam_count=selected_beam_count,
                    selected_valid_beam_count=selected_valid_beam_count,
                    debug=debug,
                    wall_pair_name=wall_pair.name,
                    region_name=region_name,
                    beam_mode="projected_xy_with_lidar_yaw",
                    selected_beams=selected_beams,
                    yaw_in_corner_deg=yaw_corner_deg,
                )
            y_corner = y_value

        x_map, y_map, yaw_map_deg = transform_pose_2d(
            wall_pair.corner_x,
            wall_pair.corner_y,
            wall_pair.corner_yaw_deg,
            x_corner,
            y_corner,
            yaw_corner_deg,
        )
        delta_local_x = x_corner - coarse_corner_x
        delta_local_y = y_corner - coarse_corner_y
        delta_local_xy_norm_m = math.hypot(delta_local_x, delta_local_y)
        delta_x = x_map - coarse.x
        delta_y = y_map - coarse.y
        delta_xy_norm_m = math.hypot(delta_x, delta_y)
        delta_yaw_deg = 0.0
        debug = self._merge_special_solver_debug_map(debug, solve_debug)
        debug = self._with_solver_debug(
            debug,
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
                "delta_local_x_m": delta_local_x,
                "delta_local_y_m": delta_local_y,
                "delta_local_xy_norm_m": delta_local_xy_norm_m,
                "delta_yaw_deg": delta_yaw_deg,
                "max_correction_xy_m": max(max_x_correction_m, max_y_correction_m),
                "max_correction_yaw_deg": max_correction_yaw_deg,
            },
        )
        if "x" in solve_axes and abs(delta_local_x) > max_x_correction_m:
            debug = self._with_special_solver_debug(
                debug,
                failure_reason="PROJECTED_X_CORRECTION_EXCEEDS_LIMIT",
                max_x_correction_m=max_x_correction_m,
            )
            return self._make_coarse_result(
                coarse,
                reason="PROJECTED_AXIS_CORRECTION_EXCEEDS_LIMIT",
                valid_beam_count=selected_valid_beam_count,
                prior_age_ms=prior_age_ms,
                usable_sensor_count=usable_sensor_count,
                selected_beam_count=selected_beam_count,
                selected_valid_beam_count=selected_valid_beam_count,
                debug=debug,
                wall_pair_name=wall_pair.name,
                region_name=region_name,
                beam_mode="projected_xy_with_lidar_yaw",
                selected_beams=selected_beams,
                yaw_in_corner_deg=yaw_corner_deg,
            )
        if "y" in solve_axes and abs(delta_local_y) > max_y_correction_m:
            debug = self._with_special_solver_debug(
                debug,
                failure_reason="PROJECTED_Y_CORRECTION_EXCEEDS_LIMIT",
                max_y_correction_m=max_y_correction_m,
            )
            return self._make_coarse_result(
                coarse,
                reason="PROJECTED_AXIS_CORRECTION_EXCEEDS_LIMIT",
                valid_beam_count=selected_valid_beam_count,
                prior_age_ms=prior_age_ms,
                usable_sensor_count=usable_sensor_count,
                selected_beam_count=selected_beam_count,
                selected_valid_beam_count=selected_valid_beam_count,
                debug=debug,
                wall_pair_name=wall_pair.name,
                region_name=region_name,
                beam_mode="projected_xy_with_lidar_yaw",
                selected_beams=selected_beams,
                yaw_in_corner_deg=yaw_corner_deg,
            )

        residual_m, target_hits = self._evaluate_projected_solution(
            pose_x=x_map,
            pose_y=y_map,
            pose_yaw_deg=yaw_map_deg,
            range_frame=range_frame,
            wall_pair=wall_pair,
            x_beam=x_beam,
            y_beam=y_beam,
            solver_cfg=solver_cfg,
        )
        residual_thresh_m = float(solver_cfg.get("residual_thresh_m", 0.03))
        score = max(0.0, 1.0 - residual_m / max(residual_thresh_m, 1e-6))
        debug = self._with_solver_debug(
            debug,
            residual_debug={
                "mean_residual_m": residual_m,
                "target_hit_count": target_hits,
                "residual_thresh_m": residual_thresh_m,
                "min_valid_corner_beams": float(selected_valid_beam_count),
                "validation_gates_block_pose": 0.0,
                "would_fail_target_hit_gate": 0.0,
                "would_fail_residual_gate": (
                    1.0 if residual_m > residual_thresh_m else 0.0
                ),
            },
        )
        if len(solve_axes) == 1:
            publish_fields = self._single_axis_publish_fields(
                x=x_map if solve_axes[0] == "x" else None,
                y=y_map if solve_axes[0] == "y" else None,
            )
        else:
            publish_fields = self._full_pose_publish_fields(
                coarse,
                x=x_map,
                y=y_map,
                yaw_deg=None,
            )
        return self._make_result(
            state=STATE_REFINED,
            reason="OK",
            pose_source="projected_xy_with_lidar_yaw",
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
            beam_mode="projected_xy_with_lidar_yaw",
            selected_beams=selected_beams,
            yaw_in_corner_deg=yaw_corner_deg,
            **publish_fields,
        )

    def _iterate_compensated_theta(
        self,
        *,
        side_beam_role: str,
        side_front_range: float,
        side_rear_range: float,
        pair_spacing_m: float,
        compensation_x0_m: float,
        theta_seed_deg: float,
        max_iterations: int,
        theta_tolerance_deg: float,
    ) -> Tuple[Optional[float], Optional[float], int, bool, Optional[str]]:
        if side_beam_role not in {"left", "right"}:
            return None, None, 0, False, "UNSUPPORTED_SIDE_BEAM_ROLE"
        if not math.isfinite(float(side_front_range)) or not math.isfinite(
            float(side_rear_range)
        ):
            return None, None, 0, False, "SIDE_RANGE_NON_FINITE"
        theta_deg = float(theta_seed_deg)
        corrected_side_front_range: Optional[float] = None
        converged = False
        iterations_used = 0
        for iteration in range(1, max_iterations + 1):
            iterations_used = iteration
            cos_theta = math.cos(math.radians(theta_deg))
            if abs(cos_theta) <= 1e-6:
                return None, None, iteration - 1, False, "THETA_COMPENSATION_SINGULAR"
            corrected_side_front_range = (
                float(side_front_range) + compensation_x0_m / cos_theta
            )
            numerator = self._theta_numerator_for_side_beam(
                side_beam_role=side_beam_role,
                corrected_side_front_range=corrected_side_front_range,
                side_rear_range=side_rear_range,
            )
            theta_next_deg = math.degrees(math.atan2(numerator, pair_spacing_m))
            if not math.isfinite(theta_next_deg):
                return None, None, iteration, False, "THETA_NON_FINITE"
            if abs(theta_next_deg - theta_deg) <= theta_tolerance_deg:
                theta_deg = theta_next_deg
                converged = True
                break
            theta_deg = theta_next_deg
        cos_theta = math.cos(math.radians(theta_deg))
        if abs(cos_theta) <= 1e-6:
            return (
                None,
                None,
                iterations_used,
                converged,
                "THETA_COMPENSATION_SINGULAR",
            )
        corrected_side_front_range = float(side_front_range) + compensation_x0_m / cos_theta
        return theta_deg, corrected_side_front_range, iterations_used, converged, None

    def _compensate_side_front_range_for_yaw(
        self,
        *,
        side_front_range: float,
        compensation_x0_m: float,
        yaw_corner_deg: float,
    ) -> Tuple[Optional[float], Optional[str]]:
        cos_theta = math.cos(math.radians(yaw_corner_deg))
        if abs(cos_theta) <= 1e-6:
            return None, "THETA_COMPENSATION_SINGULAR"
        corrected_side_front_range = (
            float(side_front_range) + compensation_x0_m / cos_theta
        )
        if not math.isfinite(corrected_side_front_range):
            return None, "CORRECTED_SIDE_FRONT_RANGE_NON_FINITE"
        return corrected_side_front_range, None

    def _theta_numerator_for_side_beam(
        self,
        *,
        side_beam_role: str,
        corrected_side_front_range: float,
        side_rear_range: float,
    ) -> float:
        if side_beam_role == "left":
            return float(side_rear_range) - float(corrected_side_front_range)
        return float(corrected_side_front_range) - float(side_rear_range)

    def _with_special_solver_debug(
        self,
        debug: Optional[Dict[str, Any]],
        **fields: Any,
    ) -> Dict[str, Any]:
        merged = dict(debug or {})
        solver_debug = dict(merged.get("solver_debug", {}))
        special_solver_debug = dict(solver_debug.get("special_solver_debug", {}))
        for key, value in fields.items():
            if isinstance(value, bool) or value is None or isinstance(value, str):
                special_solver_debug[key] = value
            elif isinstance(value, int):
                special_solver_debug[key] = int(value)
            else:
                special_solver_debug[key] = self._debug_float(float(value))
        solver_debug["special_solver_debug"] = special_solver_debug
        merged["solver_debug"] = solver_debug
        return merged

    def _with_xy_yaw_debug(
        self,
        debug: Optional[Dict[str, Any]],
        *,
        xy_yaw_source: str,
        side_laser_yaw_in_corner_deg: Optional[float],
        lidar_yaw_in_corner_deg: Optional[float],
        xy_projection_yaw_in_corner_deg: Optional[float],
    ) -> Dict[str, Any]:
        merged = dict(debug or {})
        solver_debug = dict(merged.get("solver_debug", {}))
        solver_debug["xy_yaw_source"] = xy_yaw_source
        solver_debug["side_laser_yaw_in_corner_deg"] = self._debug_float(
            side_laser_yaw_in_corner_deg
        )
        solver_debug["lidar_yaw_in_corner_deg"] = self._debug_float(
            lidar_yaw_in_corner_deg
        )
        solver_debug["xy_projection_yaw_in_corner_deg"] = self._debug_float(
            xy_projection_yaw_in_corner_deg
        )
        merged["solver_debug"] = solver_debug
        return merged

    def _merge_special_solver_debug_map(
        self,
        debug: Optional[Dict[str, Any]],
        fields: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = dict(debug or {})
        solver_debug = dict(merged.get("solver_debug", {}))
        special_solver_debug = dict(solver_debug.get("special_solver_debug", {}))
        special_solver_debug.update(fields)
        solver_debug["special_solver_debug"] = special_solver_debug
        merged["solver_debug"] = solver_debug
        return merged

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
        measured_range_overrides: Optional[Dict[str, float]] = None,
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
            measured_range = (
                measured_range_overrides[beam_name]
                if measured_range_overrides is not None
                and beam_name in measured_range_overrides
                else range_frame.ranges[beam_name]
            )
            hit_x, hit_y = self._beam_hit_point(
                pose_x=pose_x,
                pose_y=pose_y,
                pose_yaw_deg=pose_yaw_deg,
                sensor_name=beam_name,
                measured_range=measured_range,
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

    def _evaluate_projected_solution(
        self,
        pose_x: float,
        pose_y: float,
        pose_yaw_deg: float,
        range_frame: RangeFrame,
        wall_pair: WallPair,
        x_beam: Optional[str],
        y_beam: Optional[str],
        solver_cfg: Dict[str, Any],
    ) -> Tuple[float, int]:
        wall_hit_tolerance_m = float(solver_cfg.get("wall_hit_tolerance_m", 0.05))
        wall_extent_margin_m = float(solver_cfg.get("wall_extent_margin_m", 0.08))
        checks: List[Tuple[str, WallSegment]] = []
        if x_beam is not None and range_frame.valid.get(x_beam, False):
            checks.append((x_beam, self.walls[wall_pair.x_wall_name]))
        if y_beam is not None and range_frame.valid.get(y_beam, False):
            checks.append((y_beam, self.walls[wall_pair.side_wall_name]))
        if not checks:
            return float("inf"), 0

        residuals: List[float] = []
        target_hits = 0
        for beam_name, target_wall in checks:
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

    def _world_pose_to_corner_local(
        self,
        *,
        pose_x: float,
        pose_y: float,
        pose_yaw_deg: float,
        wall_pair: WallPair,
    ) -> Tuple[float, float, float]:
        world_dx = float(pose_x) - wall_pair.corner_x
        world_dy = float(pose_y) - wall_pair.corner_y
        local_x, local_y = rotate_2d(world_dx, world_dy, -wall_pair.corner_yaw_deg)
        local_yaw_deg = wrap_deg(float(pose_yaw_deg) - wall_pair.corner_yaw_deg)
        return local_x, local_y, local_yaw_deg

    def _solve_projected_corner_axis_from_beam(
        self,
        *,
        beam_name: str,
        target_wall_name: str,
        target_wall_role: str,
        pose_yaw_deg: float,
        measured_range: float,
        axis: str,
        min_dir_component_abs: float,
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        mount = self.sensor_mounts[beam_name]
        origin_dx, origin_dy = rotate_2d(mount.pos_x, mount.pos_y, pose_yaw_deg)
        dir_dx, dir_dy = rotate_2d(mount.dir_x, mount.dir_y, pose_yaw_deg)
        beam_debug = {
            "beam_name": beam_name,
            "target_wall_name": target_wall_name,
            "target_wall_role": target_wall_role,
            "range_m": self._debug_float(float(measured_range)),
            "origin_dx_m": self._debug_float(origin_dx),
            "origin_dy_m": self._debug_float(origin_dy),
            "dir_dx": self._debug_float(dir_dx),
            "dir_dy": self._debug_float(dir_dy),
            "axis": axis,
            "yaw_in_corner_deg": self._debug_float(pose_yaw_deg),
        }
        if not math.isfinite(float(measured_range)):
            beam_debug["failure_reason"] = "RANGE_NON_FINITE"
            return None, beam_debug
        if axis == "x":
            if abs(dir_dx) < min_dir_component_abs:
                beam_debug["failure_reason"] = "DIR_DX_TOO_SMALL"
                beam_debug["dir_component_abs_min"] = self._debug_float(
                    min_dir_component_abs
                )
                return None, beam_debug
            axis_value = -(origin_dx + float(measured_range) * dir_dx)
        else:
            if abs(dir_dy) < min_dir_component_abs:
                beam_debug["failure_reason"] = "DIR_DY_TOO_SMALL"
                beam_debug["dir_component_abs_min"] = self._debug_float(
                    min_dir_component_abs
                )
                return None, beam_debug
            axis_value = -(origin_dy + float(measured_range) * dir_dy)
        beam_debug["axis_value"] = self._debug_float(axis_value)
        return axis_value, beam_debug

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
