from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional

from rclpy.node import Node
from std_msgs.msg import String, UInt8

from .common import INFRARED_QUERY_COMMAND, crc16_modbus
from .infrared import (
    InfraredConfig,
    InfraredEventProcessor,
    InfraredFrame,
    InfraredMappedEvent,
)


class InfraredReceiveLayer:
    def __init__(
        self,
        *,
        node: Node,
        config: InfraredConfig,
        query_device_ids: List[int],
        latest_coarse_x_provider: Any,
        serial_port: str,
        serial_baudrate: int,
        serial_response_timeout_sec: float,
        serial_poll_rate_hz: float,
    ) -> None:
        self._node = node
        self._logger = node.get_logger()
        self._clock = node.get_clock()
        self._config = config
        self.serial_port = serial_port
        self.serial_baudrate = int(serial_baudrate)
        self.serial_response_timeout_sec = float(serial_response_timeout_sec)
        self.serial_poll_rate_hz = float(serial_poll_rate_hz)
        self.query_device_ids = list(query_device_ids)

        self._topic_pub = node.create_publisher(UInt8, config.use_topic, 10)
        self._debug_pub = node.create_publisher(String, config.debug_topic, 10)
        self._processor = InfraredEventProcessor(
            config=config,
            latest_coarse_x_provider=latest_coarse_x_provider,
        )

        self._state_lock = threading.Lock()
        self._last_poll_cycle_monotonic = 0.0
        self._next_query_index = 0

    def start(self) -> None:
        self.reset_shared_serial_state()

    def stop(self) -> None:
        return None

    def reset_shared_serial_state(self) -> None:
        with self._state_lock:
            self._last_poll_cycle_monotonic = 0.0
            self._next_query_index = 0
            self._processor.reset()

    def snapshot_status(self) -> Dict[str, Any]:
        with self._state_lock:
            shared_event = self._processor.snapshot_shared_last_event()
            board_states = {}
            for device_id in self.query_device_ids:
                state = self._processor.snapshot_board_state(device_id)
                if state is None:
                    continue
                board_states[str(device_id)] = {
                    "synced": state.synced,
                    "sync_offset_ms": state.sync_offset_ms,
                    "last_device_timestamp_ms": state.last_device_timestamp_ms,
                }
            return {
                "query_device_ids": list(self.query_device_ids),
                "active_scene": self._config.active_scene,
                "shared_last_event": (
                    None
                    if shared_event is None
                    else {
                        "raw_byte": shared_event.raw_byte,
                        "aligned_ts_ms": shared_event.aligned_ts_ms,
                        "source_device_id": shared_event.source_device_id,
                    }
                ),
                "board_states": board_states,
            }

    def claim_next_query_device_id(self) -> Optional[int]:
        if not self.query_device_ids:
            return None
        if self.serial_poll_rate_hz <= 0.0:
            return None
        poll_interval_sec = 1.0 / self.serial_poll_rate_hz
        with self._state_lock:
            now_monotonic = time.monotonic()
            if (
                self._last_poll_cycle_monotonic > 0.0
                and now_monotonic - self._last_poll_cycle_monotonic < poll_interval_sec
            ):
                return None
            device_id = self.query_device_ids[self._next_query_index]
            self._next_query_index = (self._next_query_index + 1) % len(
                self.query_device_ids
            )
            self._last_poll_cycle_monotonic = now_monotonic
            return device_id

    def build_query_frame(self, device_id: int) -> bytes:
        payload = bytes([0x5A, 0xA5, INFRARED_QUERY_COMMAND, device_id & 0xFF])
        crc = crc16_modbus(payload)
        return payload + crc.to_bytes(2, byteorder="little")

    def handle_infrared_frame(self, frame: InfraredFrame) -> None:
        with self._state_lock:
            self._handle_frame_locked(frame)

    def _handle_frame_locked(self, frame: InfraredFrame) -> None:
        result = self._processor.process_frame(frame)
        if result.action == "published" and result.event is not None:
            self._publish_event(result.event)
            return
        if result.action == "synced":
            self._logger.info(
                f"Infrared sync established for device_id={frame.device_id}"
            )
            return
        if result.reason == "DEVICE_TIMESTAMP_ROLLBACK":
            self._logger.warn(
                f"Infrared device_id={frame.device_id} timestamp rolled back; "
                "resync required on next frame"
            )

    def _publish_event(self, event: InfraredMappedEvent) -> None:
        topic_msg = UInt8()
        topic_msg.data = int(event.mapped_byte)
        self._topic_pub.publish(topic_msg)

        debug_msg = String()
        debug_msg.data = json.dumps(
            {
                "mapped_type": event.mapped_type,
                "device_id": event.device_id,
                "raw_byte": event.raw_byte,
                "mapped_byte": event.mapped_byte,
                "aligned_ts_ms": event.aligned_ts_ms,
                "scene": event.scene,
                "x": event.x,
            },
            ensure_ascii=True,
            separators=(",", ":"),
        )
        self._debug_pub.publish(debug_msg)
