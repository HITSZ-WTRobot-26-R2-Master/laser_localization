#!/usr/bin/env python3
"""Core 6-beam localization refiner node for ROS 2 Humble."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import rclpy
import yaml
from rclpy.node import Node
from rclpy.time import Time

from .common import (
    CoarsePose,
    coerce_bool,
    default_config_path,
    euler_from_quaternion_components,
    wrap_deg,
)
from .pose_solver import PoseSolveLayer
from .result_publisher import ResultPublishLayer
from .serial_receiver import SerialReceiveLayer


class AgvPoseRefinerNode(Node):
    def __init__(self) -> None:
        super().__init__("agv_pose_refiner")

        self.declare_parameter("topics_config_path", default_config_path("topics.yaml"))
        self.declare_parameter("sensors_config_path", default_config_path("sensors.yaml"))
        self.declare_parameter(
            "solver_config_path", default_config_path("map_and_solver.yaml")
        )
        self.declare_parameter("publish_tf", True)

        # Deployment-tunable overrides (empty defaults defer to YAML values).
        self.declare_parameter("lidar_pose_topic", "")
        self.declare_parameter("laser_pose_topic", "")
        self.declare_parameter("laser_status_topic", "")
        self.declare_parameter("tf_parent_frame", "")
        self.declare_parameter("tf_child_frame", "")
        self.declare_parameter("serial_port", "")
        self.declare_parameter("serial_baudrate", 0)
        self.declare_parameter("world_frame_id", "")
        self.declare_parameter("active_scene", "")

        topics_config = self._load_yaml(self.get_parameter("topics_config_path").value)
        sensors_config = self._load_yaml(self.get_parameter("sensors_config_path").value)
        solver_config = self._load_yaml(self.get_parameter("solver_config_path").value)
        publish_tf = coerce_bool(self.get_parameter("publish_tf").value)

        self._apply_param_overrides(topics_config, solver_config)

        topics_in = topics_config.get("input_topics", {})
        topics_out = topics_config.get("output_topics", {})
        timing_cfg = topics_config.get("timing", {})
        serial_cfg = topics_config.get("serial_input", {})

        self.lidar_pose_topic = str(
            topics_in.get("lidar_pose_topic", "/localization/lidar_coarse_pose")
        )
        lidar_input_format = str(
            topics_in.get("lidar_input_format", "custom_pose_fields")
        )
        if lidar_input_format != "custom_pose_fields":
            raise RuntimeError(
                "Only input_topics.lidar_input_format=custom_pose_fields is supported, "
                f"got '{lidar_input_format}'."
            )

        self._pose_msg_type = self._load_message_class(
            str(
                topics_in.get(
                    "lidar_message_type", "your_interfaces.msg.LidarPoseStamped"
                )
            )
        )

        self.solve_layer = PoseSolveLayer(
            logger=self.get_logger(),
            sensors_config=sensors_config,
            solver_config=solver_config,
            timing_cfg=timing_cfg,
        )
        self.receive_layer = SerialReceiveLayer(
            node=self,
            sensor_mounts=self.solve_layer.sensor_mounts,
            serial_cfg=serial_cfg,
            range_buffer_ms=float(timing_cfg.get("range_buffer_ms", 300.0)),
        )
        self.publish_layer = ResultPublishLayer(
            node=self,
            pose_msg_type=self._pose_msg_type,
            refined_pose_topic=str(
                topics_out.get("refined_pose_topic", "/localization/pose")
            ),
            status_topic=str(topics_out.get("status_topic", "/localization/status")),
            world_frame_id=self.solve_layer.world_frame_id,
            tf_parent_frame=str(
                topics_out.get("tf_parent_frame", self.solve_layer.world_frame_id)
            ),
            tf_child_frame=str(topics_out.get("tf_child_frame", "base_link")),
            publish_tf=publish_tf,
        )

        self.create_subscription(
            self._pose_msg_type,
            self.lidar_pose_topic,
            self._on_lidar_custom_pose,
            30,
        )
        self.receive_layer.start()
        self.get_logger().info(
            "agv_pose_refiner started "
            f"(serial={self.receive_layer.serial_port}@{self.receive_layer.serial_baudrate}, "
            f"lidar={self.lidar_pose_topic}, scene={self.solve_layer.active_scene})"
        )

    def _load_yaml(self, path_value: str) -> Dict[str, Any]:
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise RuntimeError(f"YAML root must be a mapping: {path}")
        return data

    def _apply_param_overrides(
        self, topics_config: Dict[str, Any], solver_config: Dict[str, Any]
    ) -> None:
        """Apply ROS parameter overrides to YAML-loaded config dicts.

        Only parameters with non-default values (non-empty string / non-zero int)
        override the corresponding YAML keys.  This allows config/config.yaml to
        selectively override deployment-tunable settings.
        """

        def _override(param_name: str, config_dict: Dict[str, Any], *keys: str) -> None:
            val = self.get_parameter(param_name).value
            if param_name == "serial_baudrate":
                if val == 0:
                    return
            elif not val:
                return
            d = config_dict
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = val

        _override("lidar_pose_topic", topics_config, "input_topics", "lidar_pose_topic")
        _override("laser_pose_topic", topics_config, "output_topics", "refined_pose_topic")
        _override("laser_status_topic", topics_config, "output_topics", "status_topic")
        _override("tf_parent_frame", topics_config, "output_topics", "tf_parent_frame")
        _override("tf_child_frame", topics_config, "output_topics", "tf_child_frame")
        _override("serial_port", topics_config, "serial_input", "port")
        _override("serial_baudrate", topics_config, "serial_input", "baudrate")
        _override("world_frame_id", solver_config, "world", "frame_id")
        _override("active_scene", solver_config, "scene_manager", "active_scene")

    def _load_message_class(self, type_path: str) -> type:
        module_name, _, class_name = type_path.rpartition(".")
        if not module_name or not class_name:
            raise RuntimeError(
                f"Invalid lidar_message_type '{type_path}', expected package.msg.ClassName"
            )
        try:
            module = importlib.import_module(module_name)
            message_class = getattr(module, class_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to import lidar_message_type '{type_path}': {exc}"
            ) from exc

        required_fields = ["header", "x", "y", "z", "qx", "qy", "qz", "qw"]
        sample = message_class()
        missing = [field for field in required_fields if not hasattr(sample, field)]
        if missing:
            raise RuntimeError(
                f"Message type '{type_path}' missing required fields: {missing}"
            )
        return message_class

    def _on_lidar_custom_pose(self, msg: Any) -> None:
        coarse = self._coarse_pose_from_custom_pose(msg)
        now = self.get_clock().now()
        range_frames = self.receive_layer.snapshot_frames()
        result = self.solve_layer.refine(
            coarse=coarse,
            range_frames=range_frames,
            now=now,
        )
        self.publish_layer.publish(
            coarse,
            result,
            laser_snapshot=self.receive_layer.snapshot_status(now=now),
        )

    def _coarse_pose_from_custom_pose(self, msg: Any) -> CoarsePose:
        roll_rad, pitch_rad, yaw_rad = euler_from_quaternion_components(
            float(msg.qx),
            float(msg.qy),
            float(msg.qz),
            float(msg.qw),
        )
        return CoarsePose(
            stamp=Time.from_msg(msg.header.stamp),
            x=float(msg.x),
            y=float(msg.y),
            z=float(msg.z),
            roll_rad=roll_rad,
            pitch_rad=pitch_rad,
            yaw_deg=wrap_deg(math.degrees(yaw_rad)),
        )

    def destroy_node(self) -> bool:
        self.receive_layer.stop()
        return super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = AgvPoseRefinerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
