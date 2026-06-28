from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt8

from .common import (
    INFRARED_DATA_COMMAND,
    INFRARED_DATA_FRAME_LEN,
    INFRARED_QUERY_COMMAND,
    INFRARED_QUERY_FRAME_LEN,
    STP23L_HEADER,
    crc16_modbus,
)
from .infrared import (
    InfraredConfig,
    InfraredEventProcessor,
    InfraredFrame,
    InfraredMappedEvent,
)

try:
    import serial
    from serial import SerialException
except ImportError:  # pragma: no cover - runtime dependency
    serial = None
    SerialException = Exception


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

        self._parser_lock = threading.Lock()
        self._serial_byte_buffer = bytearray()
        self._last_poll_cycle_monotonic = 0.0
        self._serial_thread: Optional[threading.Thread] = None
        self._serial_stop_event = threading.Event()
        self._serial_port_handle: Any = None

    def start(self) -> None:
        if serial is None:
            raise RuntimeError(
                "pyserial is required for infrared serial input but is not installed."
            )
        if self._serial_thread is not None:
            return
        self._serial_thread = threading.Thread(
            target=self._serial_loop,
            name="infrared_serial",
            daemon=True,
        )
        self._serial_thread.start()

    def stop(self) -> None:
        self._serial_stop_event.set()
        if self._serial_port_handle is not None:
            try:
                self._serial_port_handle.close()
            except Exception:
                pass
        if self._serial_thread is not None and self._serial_thread.is_alive():
            self._serial_thread.join(timeout=2.0)
        self._serial_thread = None

    def snapshot_status(self) -> Dict[str, Any]:
        with self._parser_lock:
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

    def _wait_for_serial_device(self) -> None:
        device_path = Path(self.serial_port)
        while rclpy.ok() and not device_path.is_char_device():
            self._logger.warn(
                f"Waiting for infrared serial device: {self.serial_port}"
            )
            self._serial_stop_event.wait(1.0)

    def _serial_loop(self) -> None:
        while not self._serial_stop_event.is_set():
            self._wait_for_serial_device()
            if self._serial_stop_event.is_set():
                return

            try:
                handle = serial.Serial(
                    port=self.serial_port,
                    baudrate=self.serial_baudrate,
                    timeout=self.serial_response_timeout_sec,
                )
            except SerialException as exc:
                self._logger.error(
                    f"Failed to open infrared serial port {self.serial_port}: {exc}"
                )
                self._serial_stop_event.wait(1.0)
                continue
            except Exception as exc:  # pragma: no cover - defensive runtime path
                self._logger.error(
                    f"Unexpected infrared serial open failure on {self.serial_port}: {exc}"
                )
                self._serial_stop_event.wait(1.0)
                continue

            try:
                with handle:
                    self._serial_port_handle = handle
                    self._prepare_serial_handle(handle)
                    self._logger.info(
                        f"Opened infrared serial port {self.serial_port} at "
                        f"{self.serial_baudrate} bps"
                    )
                    while not self._serial_stop_event.is_set():
                        self._drain_serial_input(handle)
                        self._maybe_send_queries(handle)
                        time.sleep(0.0005)
            except SerialException as exc:
                if not self._serial_stop_event.is_set():
                    self._logger.error(
                        f"Infrared serial port {self.serial_port} I/O failure: {exc}; "
                        "requesting node shutdown for container restart"
                    )
            except Exception as exc:  # pragma: no cover - defensive runtime path
                if not self._serial_stop_event.is_set():
                    self._logger.error(f"Unexpected infrared serial failure: {exc}")
            finally:
                self._serial_port_handle = None

        if not self._serial_stop_event.is_set():
            self._logger.error(
                "Infrared serial loop exited due to unrecoverable error; "
                "calling rclpy.shutdown() for container restart"
            )
            rclpy.shutdown()

    def _prepare_serial_handle(self, handle: Any) -> None:
        with self._parser_lock:
            self._serial_byte_buffer.clear()
            self._last_poll_cycle_monotonic = 0.0
            self._processor.reset()
        for reset_name in ("reset_input_buffer", "reset_output_buffer"):
            reset_fn = getattr(handle, reset_name, None)
            if reset_fn is None:
                continue
            try:
                reset_fn()
            except Exception as exc:  # pragma: no cover - driver-specific
                self._logger.warn(
                    f"Infrared serial {reset_name} failed on {self.serial_port}: {exc}"
                )

    def _drain_serial_input(self, handle: Any) -> None:
        try:
            available = int(getattr(handle, "in_waiting", 0) or 0)
            if available <= 0:
                return
            chunk = handle.read(min(available, 256))
            if not chunk:
                return
        except SerialException:
            raise
        except Exception:
            return

        with self._parser_lock:
            self._serial_byte_buffer.extend(chunk)
            self._drain_serial_buffer_locked()

    def _maybe_send_queries(self, handle: Any) -> None:
        if not self.query_device_ids:
            self._serial_stop_event.wait(0.05)
            return
        if self.serial_poll_rate_hz <= 0.0:
            return
        poll_interval_sec = 1.0 / self.serial_poll_rate_hz
        now_monotonic = time.monotonic()
        if (
            self._last_poll_cycle_monotonic > 0.0
            and now_monotonic - self._last_poll_cycle_monotonic < poll_interval_sec
        ):
            return
        for device_id in self.query_device_ids:
            handle.write(self._build_query_frame(device_id))
        handle.flush()
        self._last_poll_cycle_monotonic = now_monotonic

    def _build_query_frame(self, device_id: int) -> bytes:
        payload = bytes([0x5A, 0xA5, INFRARED_QUERY_COMMAND, device_id & 0xFF])
        crc = crc16_modbus(payload)
        return payload + crc.to_bytes(2, byteorder="little")

    def _drain_serial_buffer_locked(self) -> None:
        while True:
            header_index = self._serial_byte_buffer.find(STP23L_HEADER)
            if header_index < 0:
                if len(self._serial_byte_buffer) > INFRARED_DATA_FRAME_LEN:
                    del self._serial_byte_buffer[:-1]
                return
            if header_index > 0:
                del self._serial_byte_buffer[:header_index]
            if len(self._serial_byte_buffer) < INFRARED_QUERY_FRAME_LEN:
                return

            command = self._serial_byte_buffer[2]
            if command == INFRARED_QUERY_COMMAND:
                if len(self._serial_byte_buffer) < INFRARED_QUERY_FRAME_LEN:
                    return
                if self._is_valid_query_echo(bytes(self._serial_byte_buffer[:6])):
                    del self._serial_byte_buffer[:INFRARED_QUERY_FRAME_LEN]
                else:
                    del self._serial_byte_buffer[:1]
                continue

            if command != INFRARED_DATA_COMMAND:
                del self._serial_byte_buffer[:1]
                continue

            if len(self._serial_byte_buffer) < INFRARED_DATA_FRAME_LEN:
                return

            frame_bytes = bytes(self._serial_byte_buffer[:INFRARED_DATA_FRAME_LEN])
            payload_crc = int.from_bytes(frame_bytes[10:12], byteorder="little")
            expected_crc = crc16_modbus(frame_bytes[:10])
            if payload_crc != expected_crc:
                self._logger.warn(
                    "Dropped infrared frame due to CRC mismatch "
                    f"(expected=0x{expected_crc:04X}, got=0x{payload_crc:04X})"
                )
                del self._serial_byte_buffer[:1]
                continue

            del self._serial_byte_buffer[:INFRARED_DATA_FRAME_LEN]
            frame = self._decode_infrared_frame(frame_bytes)
            self._handle_frame_locked(frame)

    def _is_valid_query_echo(self, frame_bytes: bytes) -> bool:
        if len(frame_bytes) != INFRARED_QUERY_FRAME_LEN:
            return False
        payload_crc = int.from_bytes(frame_bytes[4:6], byteorder="little")
        expected_crc = crc16_modbus(frame_bytes[:4])
        return payload_crc == expected_crc

    def _decode_infrared_frame(self, frame_bytes: bytes) -> InfraredFrame:
        return InfraredFrame(
            rx_stamp=self._clock.now(),
            device_id=int(frame_bytes[3]),
            report_type=int(frame_bytes[4]),
            raw_byte=int(frame_bytes[5]),
            device_timestamp_ms=int.from_bytes(frame_bytes[6:10], byteorder="little"),
        )

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
