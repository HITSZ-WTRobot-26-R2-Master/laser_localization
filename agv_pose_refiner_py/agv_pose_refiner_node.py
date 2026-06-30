#!/usr/bin/env python3
"""Core 6-beam localization refiner node for ROS 2 Humble."""

from __future__ import annotations

import importlib
import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rclpy
import yaml
from rclpy.node import Node
from rclpy.time import Time

from .common import (
    CoarsePose,
    coerce_bool,
    default_config_path,
    euler_from_quaternion_components,
    parse_serial_sensor_map,
    resolve_serial_max_range_frame_age_ms,
    resolve_query_device_ids,
    wrap_deg,
)
from .infrared import parse_infrared_config, resolve_infrared_query_device_ids
from .infrared_receiver import InfraredReceiveLayer
from .pose_solver import PoseSolveLayer
from .result_publisher import ResultPublishLayer
from .serial_receiver import SerialReceiveLayer


class AgvPoseRefinerNode(Node):
    def __init__(self) -> None:
        super().__init__("agv_pose_refiner")

        # ---- Config file paths ----------------------------------------------
        self.declare_parameter(
            "sensors_config_path", default_config_path("sensors.yaml")
        )
        self.declare_parameter(
            "solver_config_path", default_config_path("map_and_solver.yaml")
        )

        # ---- Input topic ----------------------------------------------------
        self.declare_parameter("lidar_pose_topic", "/pose/lidar")
        self.declare_parameter("lidar_input_format", "custom_pose_fields")
        self.declare_parameter("lidar_message_type", "interfaces.msg.R2Pose")

        # ---- Output topics --------------------------------------------------
        self.declare_parameter("laser_pose_topic", "/pose/laser")
        self.declare_parameter("laser_status_topic", "/laser_status")
        self.declare_parameter("tf_parent_frame", "map")
        self.declare_parameter("tf_child_frame", "base_link")
        self.declare_parameter("publish_tf", True)

        # ---- Serial ---------------------------------------------------------
        self.declare_parameter("serial_port", "/dev/laser_serial")
        self.declare_parameter("serial_baudrate", 115200)
        self.declare_parameter("serial_timeout_sec", 0.02)
        self.declare_parameter("serial_min_publish_interval_ms", 5.0)
        self.declare_parameter("serial_poll_rate_hz", 10.0)
        self.declare_parameter("serial_response_timeout_sec", 0.02)
        self.declare_parameter("serial_decode_log_enabled", True)
        self.declare_parameter("serial_decode_log_interval_ms", 0)
        self.declare_parameter("serial_expect_matching_device_id", True)
        self.declare_parameter("serial_query_device_ids", [])
        self.declare_parameter("serial_max_range_frame_age_ms", 0.0)

        # ---- Infrared serial ------------------------------------------------
        self.declare_parameter("infrared_serial_port", "/dev/infrared_serial")
        self.declare_parameter("infrared_serial_baudrate", 115200)
        self.declare_parameter("infrared_serial_response_timeout_sec", 0.02)
        self.declare_parameter("infrared_serial_poll_rate_hz", 100.0)
        self.declare_parameter("infrared_query_device_ids", [3, 4])
        self.declare_parameter("infrared_use_topic", "")
        self.declare_parameter("infrared_debug_topic", "")
        self.declare_parameter("infrared_raw_topic", "")

        # ---- Solver ---------------------------------------------------------
        self.declare_parameter("world_frame_id", "map")
        self.declare_parameter("active_scene", "mode_a")

        # ---- Load YAML configs (sensors + solver only) ----------------------
        sensors_config = self._load_yaml(
            self.get_parameter("sensors_config_path").value
        )
        solver_config = self._load_yaml(
            self.get_parameter("solver_config_path").value
        )

        # ---- Parse sensor_map from sensors.yaml -----------------------------
        sensor_map = parse_serial_sensor_map(sensors_config.get("sensor_map", {}))
        serial_poll_rate_hz = float(self.get_parameter("serial_poll_rate_hz").value)
        query_device_ids = resolve_query_device_ids(
            self.get_parameter("serial_query_device_ids").value, sensor_map
        )
        self._serial_max_range_frame_age_ms = resolve_serial_max_range_frame_age_ms(
            self.get_parameter("serial_max_range_frame_age_ms").value,
            serial_poll_rate_hz,
        )
        infrared_runtime_overrides = {
            "use_topic": self.get_parameter("infrared_use_topic").value,
            "debug_topic": self.get_parameter("infrared_debug_topic").value,
            "raw_topic": self.get_parameter("infrared_raw_topic").value,
        }
        infrared_config = parse_infrared_config(
            solver_config,
            runtime_config=infrared_runtime_overrides,
        )
        infrared_query_device_ids = resolve_infrared_query_device_ids(
            self.get_parameter("infrared_query_device_ids").value
        )
        infrared_serial_port = str(self.get_parameter("infrared_serial_port").value)
        infrared_serial_baudrate = int(
            self.get_parameter("infrared_serial_baudrate").value
        )
        infrared_serial_response_timeout_sec = float(
            self.get_parameter("infrared_serial_response_timeout_sec").value
        )
        infrared_serial_poll_rate_hz = float(
            self.get_parameter("infrared_serial_poll_rate_hz").value
        )

        # ---- Validate input format ------------------------------------------
        lidar_input_format = self.get_parameter("lidar_input_format").value
        if lidar_input_format != "custom_pose_fields":
            raise RuntimeError(
                "Only lidar_input_format=custom_pose_fields is supported, "
                f"got '{lidar_input_format}'."
            )

        lidar_message_type = self.get_parameter("lidar_message_type").value
        self._pose_msg_type = self._load_message_class(lidar_message_type)

        # ---- Build layers ---------------------------------------------------
        publish_tf = coerce_bool(self.get_parameter("publish_tf").value)
        shared_serial_port = str(self.get_parameter("serial_port").value)
        shared_serial_baudrate = int(self.get_parameter("serial_baudrate").value)

        if infrared_serial_port != shared_serial_port:
            self.get_logger().warn(
                "infrared_serial_port is ignored; infrared queries share serial_port "
                f"({shared_serial_port})"
            )
        if infrared_serial_baudrate != shared_serial_baudrate:
            self.get_logger().warn(
                "infrared_serial_baudrate is ignored; infrared queries share "
                f"serial_baudrate ({shared_serial_baudrate})"
            )

        self.solve_layer = PoseSolveLayer(
            logger=self.get_logger(),
            sensors_config=sensors_config,
            solver_config=solver_config,
        )
        self._latest_coarse_pose_lock = threading.Lock()
        self._latest_coarse_x: Optional[Tuple[float, Time]] = None
        self.infrared_layer = InfraredReceiveLayer(
            node=self,
            config=infrared_config,
            query_device_ids=infrared_query_device_ids,
            latest_coarse_x_provider=self._get_latest_coarse_x_snapshot,
            serial_port=shared_serial_port,
            serial_baudrate=shared_serial_baudrate,
            serial_response_timeout_sec=infrared_serial_response_timeout_sec,
            serial_poll_rate_hz=infrared_serial_poll_rate_hz,
        )
        self.receive_layer = SerialReceiveLayer(
            node=self,
            sensor_mounts=self.solve_layer.sensor_mounts,
            sensor_map=sensor_map,
            query_device_ids=query_device_ids,
            serial_port=shared_serial_port,
            serial_baudrate=shared_serial_baudrate,
            serial_timeout_sec=self.get_parameter("serial_timeout_sec").value,
            serial_min_publish_interval_ms=self.get_parameter(
                "serial_min_publish_interval_ms"
            ).value,
            serial_poll_rate_hz=serial_poll_rate_hz,
            serial_response_timeout_sec=self.get_parameter(
                "serial_response_timeout_sec"
            ).value,
            serial_decode_log_enabled=self.get_parameter(
                "serial_decode_log_enabled"
            ).value,
            serial_decode_log_interval_ms=self.get_parameter(
                "serial_decode_log_interval_ms"
            ).value,
            serial_expect_matching_device_id=self.get_parameter(
                "serial_expect_matching_device_id"
            ).value,
            infrared_layer=self.infrared_layer,
        )
        self.publish_layer = ResultPublishLayer(
            node=self,
            pose_msg_type=self._pose_msg_type,
            refined_pose_topic=self.get_parameter("laser_pose_topic").value,
            status_topic=self.get_parameter("laser_status_topic").value,
            world_frame_id=self.solve_layer.world_frame_id,
            tf_parent_frame=self.get_parameter("tf_parent_frame").value,
            tf_child_frame=self.get_parameter("tf_child_frame").value,
            publish_tf=publish_tf,
        )

        # ---- Subscription ---------------------------------------------------
        lidar_pose_topic = self.get_parameter("lidar_pose_topic").value
        self.create_subscription(
            self._pose_msg_type,
            lidar_pose_topic,
            self._on_lidar_custom_pose,
            30,
        )
        self.receive_layer.start()
        self.get_logger().info(
            "agv_pose_refiner started "
            f"(serial={self.receive_layer.serial_port}@{self.receive_layer.serial_baudrate}, "
            f"serial_query_ids={query_device_ids}, "
            f"serial_max_range_frame_age_ms={self._serial_max_range_frame_age_ms:.1f}, "
            f"infrared_shared_serial={self.infrared_layer.serial_port}@{self.infrared_layer.serial_baudrate}, "
            f"lidar={lidar_pose_topic}, scene={self.solve_layer.active_scene})"
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
        self._set_latest_coarse_x(coarse.x, coarse.stamp)
        now = self.get_clock().now()
        range_frame = self.receive_layer.snapshot_frame(
            now=now,
            max_age_ms=self._serial_max_range_frame_age_ms,
        )
        result = self.solve_layer.refine(
            coarse=coarse,
            range_frame=range_frame,
            now=now,
            max_range_frame_age_ms=self._serial_max_range_frame_age_ms,
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

    def _set_latest_coarse_x(self, x: float, stamp: Time) -> None:
        with self._latest_coarse_pose_lock:
            self._latest_coarse_x = (float(x), stamp)

    def _get_latest_coarse_x_snapshot(self) -> Optional[Tuple[float, Time]]:
        with self._latest_coarse_pose_lock:
            return self._latest_coarse_x

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
