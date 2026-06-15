from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

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
        pose_output_valid = self._should_publish_pose(result)
        pose_msg = self._build_output_pose_msg(coarse, result)
        self._pose_pub.publish(pose_msg)

        if self._publish_tf and result.state == STATE_REFINED:
            qx, qy, qz, qw = self._result_quaternion_components(coarse, result)
            tf_msg = TransformStamped()
            tf_msg.header.frame_id = self._tf_parent_frame
            tf_msg.header.stamp = coarse.stamp.to_msg()
            tf_msg.child_frame_id = self._tf_child_frame
            tf_msg.transform.translation.x = float(result.x)
            tf_msg.transform.translation.y = float(result.y)
            tf_msg.transform.translation.z = float(coarse.z)
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
        solver_debug.update({
            "attempted": solve_attempted,
            "success": solve_success,
            "current_solver_beams": current_solver_beams,
            "selected_beam_count": result.selected_beam_count,
            "selected_valid_beam_count": result.selected_valid_beam_count,
            "target_hit_count": result.target_hit_count,
            "usable_sensor_count": result.usable_sensor_count,
        })
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
            "laser_pose_output_reason": (
                "OK" if pose_output_valid else result.reason
            ),
            "laser_pose_output_reason_text": self._laser_pose_output_reason_text(
                result.reason,
                pose_output_valid,
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
        status_msg.data = self._format_status_text(status)
        self._status_pub.publish(status_msg)

    def _build_output_pose_msg(self, coarse: CoarsePose, result: SolveResult) -> Any:
        pose_msg = self._pose_msg_type()
        pose_msg.header.frame_id = self._world_frame_id
        pose_msg.header.stamp = coarse.stamp.to_msg()
        if hasattr(pose_msg, "source"):
            pose_msg.source = "laser"
        if hasattr(pose_msg, "string"):
            pose_msg.string = "laser"
        if self._should_publish_pose(result):
            qx, qy, qz, qw = self._result_quaternion_components(coarse, result)
            pose_msg.x = float(result.x)
            pose_msg.y = float(result.y)
            pose_msg.z = float(coarse.z)
            pose_msg.qx = qx
            pose_msg.qy = qy
            pose_msg.qz = qz
            pose_msg.qw = qw
            return pose_msg

        nan = float("nan")
        pose_msg.x = nan
        pose_msg.y = nan
        pose_msg.z = nan
        pose_msg.qx = nan
        pose_msg.qy = nan
        pose_msg.qz = nan
        pose_msg.qw = nan
        return pose_msg

    def _should_publish_pose(self, result: SolveResult) -> bool:
        return result.state == STATE_REFINED and self._has_finite_pose(result)

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

    def _format_status_text(self, status: Dict[str, Any]) -> str:
        sections = ["================ /laser_status ================"]
        used_keys: Set[str] = set()

        summary_keys = [
            "localized",
            "state",
            "pose_source",
            "laser_pose_output",
            "laser_pose_output_reason",
            "laser_pose_output_reason_text",
            "reason",
            "in_solve_region",
            "solve_attempted",
            "solve_success",
            "region_name",
            "wall_pair_name",
            "beam_mode",
            "score",
            "residual_m",
            "prior_age_ms",
            "valid_beam_count",
            "usable_sensor_count",
            "selected_beam_count",
            "selected_valid_beam_count",
            "target_hit_count",
            "yaw_in_corner_deg",
            "current_solver_beams",
            "selected_beams",
        ]
        summary = {
            key: status[key] for key in summary_keys if key in status
        }
        if summary:
            self._append_section(sections, "summary", summary)
            used_keys.update(summary.keys())

        ordered_sections = [
            "world_pose",
            "corner_pose",
            "corner_world_pose",
            "coarse_pose",
            "timing_debug",
            "laser_decoded",
            "region_debug",
            "solver_debug",
        ]
        for key in ordered_sections:
            if key not in status:
                continue
            self._append_section(sections, key, status[key])
            used_keys.add(key)

        for key, value in status.items():
            if key in used_keys:
                continue
            self._append_section(sections, key, value)

        return " || ".join(sections)

    def _append_section(self, sections: List[str], title: str, value: Any) -> None:
        parts: List[str] = []
        self._append_value_lines(parts, value, prefix="")
        body = " | ".join(parts) if parts else "<empty>"
        sections.append(f"[{title}] {body}")

    def _append_value_lines(self, parts: List[str], value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            if not value:
                parts.append(f"{prefix}<empty>" if prefix else "<empty>")
                return
            for key, item in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else key
                if isinstance(item, (dict, list)):
                    self._append_value_lines(parts, item, next_prefix)
                else:
                    parts.append(f"{next_prefix}={self._format_scalar(item)}")
            return

        if isinstance(value, list):
            if not value:
                parts.append(f"{prefix}=[]" if prefix else "[]")
                return
            for index, item in enumerate(value):
                next_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
                if isinstance(item, (dict, list)):
                    self._append_value_lines(parts, item, next_prefix)
                else:
                    parts.append(f"{next_prefix}={self._format_scalar(item)}")
            return

        parts.append(
            f"{prefix}={self._format_scalar(value)}" if prefix else self._format_scalar(value)
        )

    def _format_scalar(self, value: Any) -> str:
        if value is None:
            return "None"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            if math.isnan(value):
                return "NaN"
            if math.isinf(value):
                return "Inf" if value > 0 else "-Inf"
            rendered = f"{value:.6f}"
            return rendered.rstrip("0").rstrip(".")
        return str(value)

    def _laser_pose_output_reason_text(
        self,
        reason_code: str,
        pose_output_valid: bool,
    ) -> str:
        if pose_output_valid:
            return "laser解算成功，/laser_pose发布有效位姿"

        reason_text_by_code = {
            "NON_FINITE_LIDAR_POSE": "输入lidar位姿存在NaN或Inf，无法进行laser解算",
            "NO_SERIAL_RANGE_AVAILABLE": "当前没有可用的激光测距帧",
            "NO_REGION_MATCHED": "当前粗定位不在已配置的laser解算区域内",
            "INSUFFICIENT_VALID_BEAMS": "参与角点解算的有效激光束数量不足",
            "THETA_EXCEEDS_LIMIT": "侧向双束推算出的夹角超出允许范围",
            "SAME_WALL_CONFLICT": "解算结果与目标墙体的命中关系冲突",
            "RESIDUAL_TOO_LARGE": "解算残差过大，或命中目标墙数量不足",
        }
        return reason_text_by_code.get(reason_code, reason_code)

    def _result_quaternion_components(
        self, coarse: CoarsePose, result: SolveResult
    ) -> Tuple[float, float, float, float]:
        return quaternion_components_from_rpy(
            coarse.roll_rad,
            coarse.pitch_rad,
            math.radians(float(result.yaw_deg)),
        )
