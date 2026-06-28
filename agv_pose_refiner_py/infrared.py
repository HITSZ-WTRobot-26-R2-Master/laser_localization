from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from rclpy.time import Time

from .common import abs_time_diff_ms


@dataclass(frozen=True)
class InfraredRule:
    x_min: float
    x_max: float
    raw_bytes: Tuple[int, ...]
    mapped_type: str
    send_to_topic: int


@dataclass(frozen=True)
class InfraredConfig:
    active_scene: str
    use_topic: str
    debug_topic: str
    max_coarse_pose_age_ms: float
    scenes: Dict[str, Tuple[InfraredRule, ...]]


@dataclass(frozen=True)
class LatestCoarseXSnapshot:
    x: float
    stamp: Time


@dataclass(frozen=True)
class InfraredFrame:
    rx_stamp: Time
    device_id: int
    report_type: int
    raw_byte: int
    device_timestamp_ms: int


@dataclass(frozen=True)
class InfraredMappedEvent:
    mapped_type: str
    device_id: int
    raw_byte: int
    mapped_byte: int
    aligned_ts_ms: int
    scene: str
    x: float


@dataclass
class InfraredBoardSyncState:
    synced: bool = False
    sync_offset_ms: Optional[int] = None
    last_device_timestamp_ms: Optional[int] = None


@dataclass(frozen=True)
class SharedInfraredEvent:
    mapped_byte: int
    mapped_type: str
    aligned_ts_ms: int
    source_device_id: int


@dataclass(frozen=True)
class InfraredProcessResult:
    action: str
    reason: str
    aligned_ts_ms: Optional[int] = None
    event: Optional[InfraredMappedEvent] = None


def parse_infrared_config(solver_config: Dict[str, Any]) -> InfraredConfig:
    scene_manager = solver_config.get("scene_manager", {})
    if not isinstance(scene_manager, dict):
        raise RuntimeError("scene_manager must be a mapping")
    active_scene = str(scene_manager.get("active_scene", "")).strip()
    if not active_scene:
        raise RuntimeError("scene_manager.active_scene must be configured")

    infrared_cfg = solver_config.get("infrared", {})
    if not isinstance(infrared_cfg, dict):
        raise RuntimeError("infrared must be a mapping")

    use_topic = str(infrared_cfg.get("use_topic", "")).strip()
    debug_topic = str(infrared_cfg.get("debug_topic", "")).strip()
    if not use_topic:
        raise RuntimeError("infrared.use_topic must be configured")
    if not debug_topic:
        raise RuntimeError("infrared.debug_topic must be configured")

    max_coarse_pose_age_ms = float(
        infrared_cfg.get("max_coarse_pose_age_ms", 500.0)
    )
    if max_coarse_pose_age_ms < 0.0:
        raise RuntimeError("infrared.max_coarse_pose_age_ms must be >= 0")

    scenes_cfg = infrared_cfg.get("scenes", {})
    if not isinstance(scenes_cfg, dict):
        raise RuntimeError("infrared.scenes must be a mapping")

    parsed_scenes: Dict[str, Tuple[InfraredRule, ...]] = {}
    for scene_name, scene_cfg in scenes_cfg.items():
        if not isinstance(scene_cfg, dict):
            raise RuntimeError(f"infrared.scenes.{scene_name} must be a mapping")
        raw_rules = scene_cfg.get("rules", [])
        if not isinstance(raw_rules, list) or not raw_rules:
            raise RuntimeError(
                f"infrared.scenes.{scene_name}.rules must be a non-empty list"
            )
        parsed_rules: List[InfraredRule] = []
        for index, raw_rule in enumerate(raw_rules):
            if not isinstance(raw_rule, dict):
                raise RuntimeError(
                    f"infrared.scenes.{scene_name}.rules[{index}] must be a mapping"
                )
            x_range = raw_rule.get("x_range", [])
            if not isinstance(x_range, list) or len(x_range) != 2:
                raise RuntimeError(
                    f"infrared.scenes.{scene_name}.rules[{index}].x_range "
                    "must contain exactly two values"
                )
            x_min = float(x_range[0])
            x_max = float(x_range[1])
            if x_max < x_min:
                raise RuntimeError(
                    f"infrared.scenes.{scene_name}.rules[{index}].x_range "
                    "must be ordered as [min, max]"
                )

            raw_bytes = raw_rule.get("raw_bytes", [])
            if not isinstance(raw_bytes, list) or not raw_bytes:
                raise RuntimeError(
                    f"infrared.scenes.{scene_name}.rules[{index}].raw_bytes "
                    "must be a non-empty list"
                )
            parsed_raw_bytes = tuple(
                _parse_byte_value(
                    value,
                    f"infrared.scenes.{scene_name}.rules[{index}].raw_bytes",
                )
                for value in raw_bytes
            )

            mapped_type = str(raw_rule.get("mapped_type", "")).strip()
            if not mapped_type:
                raise RuntimeError(
                    f"infrared.scenes.{scene_name}.rules[{index}].mapped_type "
                    "must be configured"
                )

            send_to_topic = _parse_byte_value(
                raw_rule.get("send_to_topic"),
                f"infrared.scenes.{scene_name}.rules[{index}].send_to_topic",
            )

            parsed_rules.append(
                InfraredRule(
                    x_min=x_min,
                    x_max=x_max,
                    raw_bytes=parsed_raw_bytes,
                    mapped_type=mapped_type,
                    send_to_topic=send_to_topic,
                )
            )
        parsed_scenes[str(scene_name)] = tuple(parsed_rules)

    if active_scene not in parsed_scenes:
        raise RuntimeError(
            f"infrared.scenes does not define active scene '{active_scene}'"
        )

    return InfraredConfig(
        active_scene=active_scene,
        use_topic=use_topic,
        debug_topic=debug_topic,
        max_coarse_pose_age_ms=max_coarse_pose_age_ms,
        scenes=parsed_scenes,
    )


def resolve_infrared_query_device_ids(configured: Optional[List[int]]) -> List[int]:
    if configured is None:
        raise RuntimeError("infrared_query_device_ids must be configured")
    if not configured:
        raise RuntimeError("infrared_query_device_ids must be a non-empty list")
    resolved: List[int] = []
    for raw_device_id in configured:
        device_id = int(raw_device_id)
        if device_id < 0 or device_id > 255:
            raise RuntimeError(
                f"infrared_query_device_ids contains invalid device id {device_id}"
            )
        resolved.append(device_id)
    return resolved


class InfraredEventProcessor:
    def __init__(
        self,
        *,
        config: InfraredConfig,
        latest_coarse_x_provider: Callable[[], Optional[Tuple[float, Time]]],
    ) -> None:
        self._config = config
        self._latest_coarse_x_provider = latest_coarse_x_provider
        self._rules = config.scenes[config.active_scene]
        self._board_states: Dict[int, InfraredBoardSyncState] = {}
        self._shared_last_event: Optional[SharedInfraredEvent] = None

    def reset(self) -> None:
        self._board_states.clear()
        self._shared_last_event = None

    def process_frame(self, frame: InfraredFrame) -> InfraredProcessResult:
        state = self._board_states.setdefault(frame.device_id, InfraredBoardSyncState())
        if (
            state.synced
            and state.last_device_timestamp_ms is not None
            and frame.device_timestamp_ms < state.last_device_timestamp_ms
        ):
            state.synced = False
            state.sync_offset_ms = None
            state.last_device_timestamp_ms = None
            return InfraredProcessResult(
                action="dropped",
                reason="DEVICE_TIMESTAMP_ROLLBACK",
            )

        if not state.synced:
            state.sync_offset_ms = self._rx_stamp_to_ms(frame.rx_stamp) - frame.device_timestamp_ms
            state.last_device_timestamp_ms = frame.device_timestamp_ms
            state.synced = True
            return InfraredProcessResult(
                action="synced",
                reason="SYNC_ESTABLISHED",
            )

        if state.sync_offset_ms is None:
            state.synced = False
            state.last_device_timestamp_ms = None
            return InfraredProcessResult(
                action="dropped",
                reason="SYNC_OFFSET_MISSING",
            )

        aligned_ts_ms = frame.device_timestamp_ms + state.sync_offset_ms
        state.last_device_timestamp_ms = frame.device_timestamp_ms

        if frame.raw_byte == 0x00:
            return InfraredProcessResult(
                action="dropped",
                reason="RAW_BYTE_ZERO",
                aligned_ts_ms=aligned_ts_ms,
            )

        if (
            self._shared_last_event is not None
            and aligned_ts_ms <= self._shared_last_event.aligned_ts_ms
        ):
            return InfraredProcessResult(
                action="dropped",
                reason="ALIGNED_TIMESTAMP_NOT_NEWER",
                aligned_ts_ms=aligned_ts_ms,
            )

        coarse_snapshot = self._snapshot_latest_coarse_x()
        if coarse_snapshot is None:
            return InfraredProcessResult(
                action="dropped",
                reason="NO_COARSE_X",
                aligned_ts_ms=aligned_ts_ms,
            )

        coarse_x_age_ms = abs_time_diff_ms(frame.rx_stamp, coarse_snapshot.stamp)
        if coarse_x_age_ms > self._config.max_coarse_pose_age_ms:
            return InfraredProcessResult(
                action="dropped",
                reason="COARSE_X_TOO_OLD",
                aligned_ts_ms=aligned_ts_ms,
            )

        matched_rule = self._match_rule(coarse_snapshot.x, frame.raw_byte)
        if matched_rule is None:
            return InfraredProcessResult(
                action="dropped",
                reason="NO_RULE_MATCH",
                aligned_ts_ms=aligned_ts_ms,
            )

        if (
            self._shared_last_event is not None
            and matched_rule.send_to_topic == self._shared_last_event.mapped_byte
            and matched_rule.mapped_type == self._shared_last_event.mapped_type
        ):
            return InfraredProcessResult(
                action="dropped",
                reason="MAPPED_EVENT_DUPLICATED",
                aligned_ts_ms=aligned_ts_ms,
            )

        self._shared_last_event = SharedInfraredEvent(
            mapped_byte=matched_rule.send_to_topic,
            mapped_type=matched_rule.mapped_type,
            aligned_ts_ms=aligned_ts_ms,
            source_device_id=frame.device_id,
        )

        return InfraredProcessResult(
            action="published",
            reason="MAPPED",
            aligned_ts_ms=aligned_ts_ms,
            event=InfraredMappedEvent(
                mapped_type=matched_rule.mapped_type,
                device_id=frame.device_id,
                raw_byte=frame.raw_byte,
                mapped_byte=matched_rule.send_to_topic,
                aligned_ts_ms=aligned_ts_ms,
                scene=self._config.active_scene,
                x=coarse_snapshot.x,
            ),
        )

    def snapshot_shared_last_event(self) -> Optional[SharedInfraredEvent]:
        return self._shared_last_event

    def snapshot_board_state(self, device_id: int) -> Optional[InfraredBoardSyncState]:
        state = self._board_states.get(device_id)
        if state is None:
            return None
        return InfraredBoardSyncState(
            synced=state.synced,
            sync_offset_ms=state.sync_offset_ms,
            last_device_timestamp_ms=state.last_device_timestamp_ms,
        )

    def _snapshot_latest_coarse_x(self) -> Optional[LatestCoarseXSnapshot]:
        snapshot = self._latest_coarse_x_provider()
        if snapshot is None:
            return None
        x, stamp = snapshot
        return LatestCoarseXSnapshot(x=float(x), stamp=stamp)

    def _match_rule(self, coarse_x: float, raw_byte: int) -> Optional[InfraredRule]:
        for rule in self._rules:
            if rule.x_min <= coarse_x <= rule.x_max and raw_byte in rule.raw_bytes:
                return rule
        return None

    def _rx_stamp_to_ms(self, stamp: Time) -> int:
        return int(stamp.nanoseconds / 1e6)


def _parse_byte_value(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{field_name} must be an integer byte value")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        parsed = int(value, 0)
    else:
        raise RuntimeError(f"{field_name} must be an integer byte value")
    if parsed < 0 or parsed > 0xFF:
        raise RuntimeError(f"{field_name} must be in [0x00, 0xFF], got {parsed}")
    return parsed
