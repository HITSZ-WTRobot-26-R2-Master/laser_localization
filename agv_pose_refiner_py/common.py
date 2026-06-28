from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from rclpy.time import Time


SENSOR_ORDER = [
    "front_center",
    "rear_center",
    "left_front",
    "left_rear",
    "right_front",
    "right_rear",
]

STP23L_HEADER = b"\x5A\xA5"
STP23L_QUERY_COMMAND = 0x01
STP23L_DATA_COMMAND = 0x81
STP23L_QUERY_FRAME_LEN = 6
STP23L_DATA_FRAME_LEN = 16
STP23L_INVALID_DISTANCE_MM = 0xFFFF
INFRARED_QUERY_COMMAND = 0x02
INFRARED_DATA_COMMAND = 0x82
INFRARED_QUERY_FRAME_LEN = 6
INFRARED_DATA_FRAME_LEN = 12

STATE_REFINED = "REFINED"
STATE_COARSE_ONLY = "COARSE_ONLY"
STATE_CANNOT_LOCALIZE = "CANNOT_LOCALIZE"


@dataclass
class SensorMount:
    pos_x: float
    pos_y: float
    dir_x: float
    dir_y: float
    min_range_m: float
    max_range_m: float


@dataclass
class SensorGeometry:
    x_front: float
    x_rear: float
    y_left: float
    y_right: float
    x_left_pair: float
    x_right_pair: float


@dataclass
class RangeFrame:
    stamp: Time
    ranges: Dict[str, float]
    valid: Dict[str, bool]


@dataclass
class CoarsePose:
    stamp: Time
    x: float
    y: float
    z: float
    roll_rad: float
    pitch_rad: float
    yaw_deg: float


@dataclass
class WallSegment:
    name: str
    orientation: str
    const_value: float
    min_axis: float
    max_axis: float


@dataclass
class WallPair:
    name: str
    x_wall_name: str
    x_wall_role: str
    side_wall_name: str
    side_wall_role: str
    corner_x: float
    corner_y: float
    corner_yaw_deg: float


@dataclass
class RegionMatch:
    name: str
    wall_pairs: List[WallPair]
    region_config: Optional[Dict[str, Any]] = None


@dataclass
class BeamSelection:
    x_beam: str
    x_beam_role: str
    x_offset_m: float
    side_front_beam: str
    side_rear_beam: str
    side_beam_role: str
    side_offset_m: float
    pair_spacing_m: float
    yaw_in_corner_deg: float
    beam_mode: str

    def required_beams(self) -> List[str]:
        return [self.x_beam, self.side_front_beam, self.side_rear_beam]


@dataclass
class SolveResult:
    state: str
    reason: str
    pose_source: str
    localized: bool
    x: Optional[float]
    y: Optional[float]
    yaw_deg: Optional[float]
    valid_beam_count: int
    score: float
    prior_age_ms: Optional[float]
    usable_sensor_count: int = 0
    selected_beam_count: int = 0
    selected_valid_beam_count: int = 0
    target_hit_count: int = 0
    debug: Optional[Dict[str, Any]] = None
    residual_m: Optional[float] = None
    wall_pair_name: Optional[str] = None
    region_name: Optional[str] = None
    beam_mode: Optional[str] = None
    selected_beams: Optional[List[str]] = None
    yaw_in_corner_deg: Optional[float] = None
    publish_x: Optional[float] = None
    publish_y: Optional[float] = None
    publish_z: Optional[float] = None
    publish_yaw_deg: Optional[float] = None


@dataclass
class BoardFrame:
    rx_stamp: Time
    device_id: int
    report_type: int
    status_bits: int
    distances_mm: List[int]


@dataclass
class SerialSensorMapping:
    logical_sensor: str
    device_id: int
    slot_index: int


def parse_serial_sensor_map(
    raw_config: Dict[str, Any],
) -> Dict[str, SerialSensorMapping]:
    missing = [name for name in SENSOR_ORDER if name not in raw_config]
    if missing:
        raise RuntimeError(f"sensor_map missing keys: {missing}")

    parsed: Dict[str, SerialSensorMapping] = {}
    used_pairs = set()
    for logical_sensor in SENSOR_ORDER:
        item = raw_config[logical_sensor]
        device_id = int(item["device_id"])
        slot_index = int(item["slot_index"])
        if slot_index < 0 or slot_index > 3:
            raise RuntimeError(
                f"{logical_sensor} slot_index must be in [0, 3], got {slot_index}"
            )
        pair = (device_id, slot_index)
        if pair in used_pairs:
            raise RuntimeError(f"Duplicate serial sensor mapping for {pair}")
        used_pairs.add(pair)
        parsed[logical_sensor] = SerialSensorMapping(
            logical_sensor=logical_sensor,
            device_id=device_id,
            slot_index=slot_index,
        )
    return parsed


def resolve_query_device_ids(
    configured: Optional[List[int]],
    sensor_map: Dict[str, SerialSensorMapping],
) -> List[int]:
    if configured is not None:
        if not configured:
            raise RuntimeError("serial_query_device_ids must be a non-empty list")
        for device_id in configured:
            if device_id < 0 or device_id > 255:
                raise RuntimeError(
                    f"serial_query_device_ids contains invalid device id {device_id}"
                )
        return [int(d) for d in configured]
    return sorted({mapping.device_id for mapping in sensor_map.values()})


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def wrap_deg(angle_deg: float) -> float:
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return angle_deg


def euler_from_quaternion_components(
    qx: float, qy: float, qz: float, qw: float
) -> Tuple[float, float, float]:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def quaternion_components_from_rpy(
    roll_rad: float, pitch_rad: float, yaw_rad: float
) -> Tuple[float, float, float, float]:
    cy = math.cos(yaw_rad * 0.5)
    sy = math.sin(yaw_rad * 0.5)
    cp = math.cos(pitch_rad * 0.5)
    sp = math.sin(pitch_rad * 0.5)
    cr = math.cos(roll_rad * 0.5)
    sr = math.sin(roll_rad * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def time_diff_ms(lhs: Time, rhs: Time) -> float:
    return (lhs.nanoseconds - rhs.nanoseconds) / 1e6


def abs_time_diff_ms(lhs: Time, rhs: Time) -> float:
    return abs(time_diff_ms(lhs, rhs))


def rotate_2d(x: float, y: float, yaw_deg: float) -> Tuple[float, float]:
    yaw = math.radians(yaw_deg)
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (c * x - s * y, s * x + c * y)


def transform_pose_2d(
    ref_x: float,
    ref_y: float,
    ref_yaw_deg: float,
    local_x: float,
    local_y: float,
    local_yaw_deg: float,
) -> Tuple[float, float, float]:
    world_dx, world_dy = rotate_2d(local_x, local_y, ref_yaw_deg)
    return (
        ref_x + world_dx,
        ref_y + world_dy,
        wrap_deg(local_yaw_deg + ref_yaw_deg),
    )


def crc16_modbus(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def default_config_path(filename: str) -> str:
    config_roots: List[Path] = []

    env_dir = os.environ.get("AGV_POSE_REFINER_CONFIG_DIR")
    if env_dir:
        config_roots.append(Path(env_dir))

    try:
        share_dir = Path(get_package_share_directory("agv_pose_refiner"))
        config_roots.append(share_dir / "config")
    except PackageNotFoundError:
        pass

    config_roots.append(Path(__file__).resolve().parents[1] / "config")

    for root in config_roots:
        candidate = root / filename
        if candidate.exists():
            return str(candidate)

    return str(config_roots[-1] / filename)
