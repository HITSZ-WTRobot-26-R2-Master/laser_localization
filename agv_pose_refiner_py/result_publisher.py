from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

from .common import (
    STATE_REFINED,
    CoarsePose,
    SolveResult,
    quaternion_components_from_rpy,
)


class ResultPublishLayer:
    def __init__(
        self,
        node: Any,
        pose_msg_type: type,
        refined_pose_topic: str,
        status_topic: str,
        world_frame_id: str,
        tf_parent_frame: str,
        tf_child_frame: str,
        publish_tf: bool,
    ) -> None:
        self._pose_msg_type = pose_msg_type
        self._world_frame_id = world_frame_id
        self._tf_parent_frame = tf_parent_frame
        self._tf_child_frame = tf_child_frame
        self._publish_tf = publish_tf

        self._pose_pub = node.create_publisher(pose_msg_type, refined_pose_topic, 10)
        self._status_pub = node.create_publisher(String, status_topic, 10)
        self._tf_broadcaster = TransformBroadcaster(node)

    def publish(
        self,
        coarse: CoarsePose,
        result: SolveResult,
        laser_snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        pose_output_fields = self._output_valid_fields(result)
        pose_output_valid = bool(pose_output_fields)
        pose_output_complete = self._should_publish_tf(result)
        pose_msg = self._build_output_pose_msg(coarse, result)
        self._pose_pub.publish(pose_msg)

        if self._publish_tf and self._should_publish_tf(result):
            qx, qy, qz, qw = self._result_quaternion_components(coarse, result)
            tf_msg = TransformStamped()
            tf_msg.header.frame_id = self._tf_parent_frame
            tf_msg.header.stamp = coarse.stamp.to_msg()
            tf_msg.child_frame_id = self._tf_child_frame
            tf_msg.transform.translation.x = float(result.publish_x)
            tf_msg.transform.translation.y = float(result.publish_y)
            tf_msg.transform.translation.z = float(result.publish_z)
            tf_msg.transform.rotation.x = qx
            tf_msg.transform.rotation.y = qy
            tf_msg.transform.rotation.z = qz
            tf_msg.transform.rotation.w = qw
            self._tf_broadcaster.sendTransform(tf_msg)

        debug = result.debug or {}
        coarse_pose_debug = debug.get(
            "coarse_pose",
            {
                "x": float(coarse.x),
                "y": float(coarse.y),
                "z": float(coarse.z),
                "yaw_deg": float(coarse.yaw_deg),
                "roll_rad": float(coarse.roll_rad),
                "pitch_rad": float(coarse.pitch_rad),
            },
        )
        timing_debug = debug.get(
            "timing_debug",
            {
                "transport_delay_ms": None,
                "range_frame_found": None,
                "range_frame_age_ms": result.prior_age_ms,
                "range_frame_count": None,
            },
        )
        region_debug = debug.get(
            "region_debug",
            {"evaluated": False, "matched": False},
        )
        in_solve_region = bool(region_debug.get("matched", False))
        solve_attempted = bool(result.selected_beams)
        solve_success = result.state == STATE_REFINED
        current_solver_beams = list(result.selected_beams or [])
        solver_debug = dict(debug.get("solver_debug", {}))
        solver_debug.update(
            {
                "attempted": solve_attempted,
                "success": solve_success,
                "current_solver_beams": current_solver_beams,
                "selected_beam_count": result.selected_beam_count,
                "selected_valid_beam_count": result.selected_valid_beam_count,
                "target_hit_count": result.target_hit_count,
                "usable_sensor_count": result.usable_sensor_count,
            }
        )
        if result.beam_mode is not None:
            solver_debug["beam_mode"] = result.beam_mode
        corner_pose = self._extract_pose_debug(solver_debug.get("corner_pose"))
        corner_world_pose = self._extract_pose_debug(
            solver_debug.get("corner_world_pose")
        )
        world_pose = self._extract_pose_debug(solver_debug.get("candidate_pose"))
        if world_pose is None and solve_success and self._has_finite_pose(result):
            world_pose = {
                "x": float(result.x),
                "y": float(result.y),
                "yaw_deg": float(result.yaw_deg),
            }

        status = {
            "localized": result.localized,
            "state": result.state,
            "pose_source": result.pose_source,
            "laser_pose_output": "POSE" if pose_output_valid else "NAN",
            "laser_pose_output_fields": pose_output_fields,
            "laser_pose_output_complete": pose_output_complete,
            "laser_pose_output_reason": ("OK" if pose_output_valid else result.reason),
            "laser_pose_output_reason_text": self._laser_pose_output_reason_text(
                result.reason,
                pose_output_valid,
                pose_output_complete,
            ),
            "in_solve_region": in_solve_region,
            "solve_attempted": solve_attempted,
            "solve_success": solve_success,
            "current_solver_beams": current_solver_beams,
            "coarse_pose": coarse_pose_debug,
            "timing_debug": timing_debug,
            "region_debug": region_debug,
            "solver_debug": solver_debug,
            "valid_beam_count": result.valid_beam_count,
            "usable_sensor_count": result.usable_sensor_count,
            "selected_beam_count": result.selected_beam_count,
            "selected_valid_beam_count": result.selected_valid_beam_count,
            "target_hit_count": result.target_hit_count,
            "score": result.score,
            "prior_age_ms": result.prior_age_ms,
            "reason": result.reason,
        }
        if laser_snapshot is not None:
            status["laser_decoded"] = laser_snapshot
        if corner_pose is not None:
            status["corner_pose"] = {
                "frame_id": "corner_local",
                "wall_pair_name": result.wall_pair_name,
                **corner_pose,
            }
        if corner_world_pose is not None:
            status["corner_world_pose"] = {
                "frame_id": self._world_frame_id,
                **corner_world_pose,
            }
        if world_pose is not None:
            status["world_pose"] = {
                "frame_id": self._world_frame_id,
                "source": result.pose_source,
                **world_pose,
            }
        if result.residual_m is not None:
            status["residual_m"] = result.residual_m
        if result.region_name is not None:
            status["region_name"] = result.region_name
        if result.wall_pair_name is not None:
            status["wall_pair_name"] = result.wall_pair_name
        if result.beam_mode is not None:
            status["beam_mode"] = result.beam_mode
        if result.selected_beams is not None:
            status["selected_beams"] = result.selected_beams
        if result.yaw_in_corner_deg is not None:
            status["yaw_in_corner_deg"] = result.yaw_in_corner_deg

        status_msg = String()
        status_msg.data = json.dumps(status, allow_nan=False, default=str)
        self._status_pub.publish(status_msg)

    def _build_output_pose_msg(self, coarse: CoarsePose, result: SolveResult) -> Any:
        pose_msg = self._pose_msg_type()
        pose_msg.header.frame_id = self._world_frame_id
        pose_msg.header.stamp = coarse.stamp.to_msg()
        if hasattr(pose_msg, "source"):
            pose_msg.source = "laser"
        if hasattr(pose_msg, "string"):
            pose_msg.string = "laser"

        pose_msg.x = self._output_float_or_nan(result.publish_x)
        pose_msg.y = self._output_float_or_nan(result.publish_y)
        pose_msg.z = self._output_float_or_nan(result.publish_z)
        if self._has_finite_output_yaw(result):
            qx, qy, qz, qw = self._result_quaternion_components(coarse, result)
            pose_msg.qx = qx
            pose_msg.qy = qy
            pose_msg.qz = qz
            pose_msg.qw = qw
            return pose_msg

        nan = float("nan")
        pose_msg.qx = nan
        pose_msg.qy = nan
        pose_msg.qz = nan
        pose_msg.qw = nan
        return pose_msg

    def _should_publish_tf(self, result: SolveResult) -> bool:
        return result.state == STATE_REFINED and self._has_finite_output_tf_pose(result)

    def _has_finite_output_tf_pose(self, result: SolveResult) -> bool:
        return (
            self._finite_output_value(result.publish_x) is not None
            and self._finite_output_value(result.publish_y) is not None
            and self._finite_output_value(result.publish_z) is not None
            and self._has_finite_output_yaw(result)
        )

    def _has_finite_output_yaw(self, result: SolveResult) -> bool:
        return self._finite_output_value(result.publish_yaw_deg) is not None

    def _output_valid_fields(self, result: SolveResult) -> List[str]:
        fields: List[str] = []
        for key, value in (
            ("x", result.publish_x),
            ("y", result.publish_y),
            ("z", result.publish_z),
        ):
            if self._finite_output_value(value) is not None:
                fields.append(key)
        if self._has_finite_output_yaw(result):
            fields.extend(["qx", "qy", "qz", "qw"])
        return fields

    def _finite_output_value(self, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        value_float = float(value)
        if not math.isfinite(value_float):
            return None
        return value_float

    def _output_float_or_nan(self, value: Optional[float]) -> float:
        finite_value = self._finite_output_value(value)
        if finite_value is None:
            return float("nan")
        return finite_value

    def _has_finite_pose(self, result: SolveResult) -> bool:
        return (
            result.x is not None
            and result.y is not None
            and result.yaw_deg is not None
            and math.isfinite(float(result.x))
            and math.isfinite(float(result.y))
            and math.isfinite(float(result.yaw_deg))
        )

    def _extract_pose_debug(self, pose_debug: Any) -> Optional[Dict[str, float]]:
        if not isinstance(pose_debug, dict):
            return None
        extracted: Dict[str, float] = {}
        for key in ("x", "y", "yaw_deg"):
            value = pose_debug.get(key)
            if value is None:
                return None
            value_float = float(value)
            if not math.isfinite(value_float):
                return None
            extracted[key] = value_float
        return extracted

    def _laser_pose_output_reason_text(
        self,
        reason_code: str,
        pose_output_valid: bool,
        pose_output_complete: bool,
    ) -> str:
        if pose_output_valid:
            if not pose_output_complete:
                return "laser solve succeeded; /laser_pose contains only valid solved fields"
            return "laser solve succeeded; /laser_pose contains a complete pose"

        reason_text_by_code = {
            "NON_FINITE_LIDAR_POSE": "input lidar pose contains NaN or Inf",
            "NO_SERIAL_RANGE_AVAILABLE": "no serial range frame is available",
            "NO_REGION_MATCHED": "coarse lidar pose is outside all configured solve regions",
            "INSUFFICIENT_VALID_BEAMS": "not enough valid beams for solving",
            "THETA_EXCEEDS_LIMIT": "solved yaw exceeds configured limit",
            "RESIDUAL_TOO_LARGE": "residual is too large",
        }
        return reason_text_by_code.get(reason_code, reason_code)

    def _result_quaternion_components(
        self, coarse: CoarsePose, result: SolveResult
    ) -> Tuple[float, float, float, float]:
        return quaternion_components_from_rpy(
            coarse.roll_rad,
            coarse.pitch_rad,
            math.radians(float(result.publish_yaw_deg)),
        )
