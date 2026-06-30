from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from .common import (
    INFRARED_DATA_COMMAND,
    INFRARED_DATA_FRAME_LEN,
    INFRARED_QUERY_COMMAND,
    INFRARED_QUERY_FRAME_LEN,
    SENSOR_ORDER,
    STP23L_DATA_COMMAND,
    STP23L_DATA_FRAME_LEN,
    STP23L_HEADER,
    STP23L_INVALID_DISTANCE_MM,
    STP23L_QUERY_COMMAND,
    STP23L_QUERY_FRAME_LEN,
    BoardFrame,
    RangeFrame,
    SensorMount,
    SerialSensorMapping,
    abs_time_diff_ms,
    crc16_modbus,
)
from .infrared import InfraredFrame

try:
    import serial
    from serial import SerialException
except ImportError:  # pragma: no cover - runtime dependency
    serial = None
    SerialException = Exception


class SerialReceiveLayer:
    def __init__(
        self,
        node: Node,
        sensor_mounts: Dict[str, SensorMount],
        sensor_map: Dict[str, SerialSensorMapping],
        query_device_ids: List[int],
        serial_port: str,
        serial_baudrate: int,
        serial_timeout_sec: float,
        serial_min_publish_interval_ms: float,
        serial_poll_rate_hz: float,
        serial_response_timeout_sec: float,
        serial_decode_log_enabled: bool,
        serial_decode_log_interval_ms: float,
        serial_expect_matching_device_id: bool,
        infrared_layer: Optional[Any] = None,
    ) -> None:
        self._node = node
        self._logger = node.get_logger()
        self._clock = node.get_clock()
        self._sensor_mounts = sensor_mounts
        self._sensor_map = sensor_map

        self.serial_port = serial_port
        self.serial_baudrate = serial_baudrate
        self.serial_timeout_sec = serial_timeout_sec
        self.serial_min_publish_interval_ms = serial_min_publish_interval_ms
        self.serial_poll_rate_hz = serial_poll_rate_hz
        self.serial_response_timeout_sec = serial_response_timeout_sec
        self.serial_decode_log_enabled = serial_decode_log_enabled
        self.serial_decode_log_interval_ms = max(0.0, float(serial_decode_log_interval_ms))
        self.serial_expect_matching_device_id = serial_expect_matching_device_id
        self.serial_query_device_ids = list(query_device_ids)
        self._infrared_layer = infrared_layer

        self._latest_range_frame: Optional[RangeFrame] = None
        self._latest_range_frame_lock = threading.Lock()
        self._latest_serial_ranges_m: Dict[str, float] = {
            name: float("nan") for name in SENSOR_ORDER
        }
        self._latest_serial_valid: Dict[str, bool] = {
            name: False for name in SENSOR_ORDER
        }
        self._last_serial_frame_stamp: Optional[Time] = None

        self._parser_lock = threading.Lock()
        self._pending_board_frames: Dict[int, BoardFrame] = {}
        self._pending_infrared_frames: Dict[int, InfraredFrame] = {}
        self._latest_board_frames: Dict[int, BoardFrame] = {}
        self._last_decode_log_stamp_ns_by_device: Dict[int, int] = {}
        self._serial_byte_buffer = bytearray()

        self._serial_thread: Optional[threading.Thread] = None
        self._serial_stop_event = threading.Event()
        self._serial_port_handle: Any = None
        self._last_poll_cycle_monotonic = 0.0

    def start(self) -> None:
        if serial is None:
            raise RuntimeError(
                "pyserial is required for serial STP23L input but is not installed."
            )
        if self._serial_thread is not None:
            return
        self._serial_thread = threading.Thread(
            target=self._serial_loop,
            name="stp23l_serial",
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
        self._clear_latest_range_state()
        self._serial_thread = None

    def snapshot_frame(self, *, now: Optional[Time] = None, max_age_ms: float = 0.0) -> Optional[RangeFrame]:
        with self._latest_range_frame_lock:
            frame = self._latest_range_frame
        if frame is None:
            return None
        if max_age_ms > 0.0:
            snapshot_now = now or self._clock.now()
            if abs_time_diff_ms(snapshot_now, frame.stamp) > max_age_ms:
                return None
        return frame

    def snapshot_status(self, now: Optional[Time] = None) -> Dict[str, Any]:
        snapshot_now = now or self._clock.now()
        with self._parser_lock:
            device_frames = {
                str(device_id): self._build_device_status_snapshot(
                    board_frame=board_frame,
                    now=snapshot_now,
                )
                for device_id, board_frame in sorted(self._latest_board_frames.items())
            }
            logical_sensors = {
                sensor_name: self._build_logical_sensor_status_snapshot(sensor_name)
                for sensor_name in SENSOR_ORDER
            }
            latest_range_frame_age_ms = None
            if self._last_serial_frame_stamp is not None:
                latest_range_frame_age_ms = abs_time_diff_ms(
                    snapshot_now, self._last_serial_frame_stamp
                )

        return {
            "has_data": bool(device_frames),
            "query_device_ids": list(self.serial_query_device_ids),
            "latest_range_frame_age_ms": latest_range_frame_age_ms,
            "device_frames": device_frames,
            "logical_sensors": logical_sensors,
        }

    # ---- Serial lifecycle ---------------------------------------------------

    def _wait_for_serial_device(self) -> None:
        device_path = Path(self.serial_port)
        while rclpy.ok() and not device_path.is_char_device():
            self._logger.warn(
                f"Waiting for serial device: {self.serial_port}"
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
                    timeout=self.serial_timeout_sec,
                )
            except SerialException as exc:
                self._logger.error(
                    f"Failed to open serial port {self.serial_port}: {exc}"
                )
                self._serial_stop_event.wait(1.0)
                continue
            except Exception as exc:  # pragma: no cover - defensive runtime path
                self._logger.error(
                    f"Unexpected serial open failure on {self.serial_port}: {exc}"
                )
                self._serial_stop_event.wait(1.0)
                continue

            try:
                with handle:
                    self._serial_port_handle = handle
                    self._prepare_serial_handle(handle)
                    self._logger.info(
                        f"Opened STP23L serial port {self.serial_port} at "
                        f"{self.serial_baudrate} bps"
                    )
                    while not self._serial_stop_event.is_set():
                        self._drain_serial_input(handle)
                        if not self._maybe_poll_cycle(handle):
                            self._maybe_poll_infrared_transaction(handle)
                        self._serial_stop_event.wait(0.0005)
            except SerialException as exc:
                if not self._serial_stop_event.is_set():
                    self._logger.error(
                        f"Serial port {self.serial_port} I/O failure: {exc}; "
                        "requesting node shutdown for container restart"
                    )
            except Exception as exc:  # pragma: no cover - defensive runtime path
                if not self._serial_stop_event.is_set():
                    self._logger.error(f"Unexpected serial failure: {exc}")
            finally:
                self._serial_port_handle = None

        if not self._serial_stop_event.is_set():
            self._logger.error(
                "Serial loop exited due to unrecoverable error; "
                "calling rclpy.shutdown() for container restart"
            )
            rclpy.shutdown()

    def _prepare_serial_handle(self, handle: Any) -> None:
        with self._parser_lock:
            self._serial_byte_buffer.clear()
            self._pending_board_frames.clear()
            self._pending_infrared_frames.clear()
            self._latest_board_frames.clear()
            self._last_decode_log_stamp_ns_by_device.clear()
            self._last_poll_cycle_monotonic = 0.0
        self._clear_latest_range_state()
        if self._infrared_layer is not None:
            self._infrared_layer.reset_shared_serial_state()
        for reset_name in ("reset_input_buffer", "reset_output_buffer"):
            reset_fn = getattr(handle, reset_name, None)
            if reset_fn is None:
                continue
            try:
                reset_fn()
            except Exception as exc:  # pragma: no cover - driver-specific
                self._logger.warn(
                    f"Serial {reset_name} failed on {self.serial_port}: {exc}"
                )

    # ---- Non-blocking read + parse ------------------------------------------

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

    # ---- Polling cycle ------------------------------------------------------

    def _maybe_poll_infrared_transaction(self, handle: Any) -> bool:
        if self._infrared_layer is None:
            return False
        device_id = self._infrared_layer.claim_next_query_device_id()
        if device_id is None:
            return False

        self._clear_pending_infrared_query_responses()
        handle.write(self._infrared_layer.build_query_frame(device_id))
        handle.flush()
        frame = self._wait_for_infrared_response_active(
            handle,
            device_id,
            self._infrared_layer.serial_response_timeout_sec,
        )
        if frame is None:
            buffered_len, pending_board_ids, pending_infrared_ids = (
                self._snapshot_rx_debug_state()
            )
            self._logger.warn(
                f"No infrared response for device_id={device_id} within "
                f"{self._infrared_layer.serial_response_timeout_sec:.3f}s "
                f"(buffered={buffered_len}B, stp23l_pending={pending_board_ids}, "
                f"infrared_pending={pending_infrared_ids})"
            )
        return True

    def _maybe_poll_cycle(self, handle: Any) -> bool:
        if not self.serial_query_device_ids:
            return False
        if self.serial_poll_rate_hz > 0.0:
            cycle_period_sec = 1.0 / self.serial_poll_rate_hz
            now_monotonic = time.monotonic()
            if (
                self._last_poll_cycle_monotonic > 0.0
                and now_monotonic - self._last_poll_cycle_monotonic < cycle_period_sec
            ):
                return False
            self._last_poll_cycle_monotonic = now_monotonic
        self._poll_cycle(handle)
        return True

    def _poll_cycle(self, handle: Any) -> None:
        self._clear_pending_query_responses()
        cycle_board_frames: Dict[int, BoardFrame] = {}

        for device_id in self.serial_query_device_ids:
            if self._serial_stop_event.is_set():
                return
            handle.write(self._build_query_frame(device_id))
            handle.flush()
            board_frame = self._wait_for_response_active(handle, device_id)
            if board_frame is None:
                buffered_len, pending_ids, _ = self._snapshot_rx_debug_state()
                self._logger.warn(
                    f"No STP23L response for device_id={device_id} within "
                    f"{self.serial_response_timeout_sec:.3f}s "
                    f"(buffered={buffered_len}B, pending={pending_ids})"
                )
                continue
            cycle_board_frames[device_id] = board_frame

        if len(cycle_board_frames) == len(self.serial_query_device_ids):
            publish_stamp = max(
                frame.rx_stamp.nanoseconds for frame in cycle_board_frames.values()
            )
            stamp = Time(nanoseconds=publish_stamp, clock_type=self._clock.clock_type)
            self._maybe_publish_cycle_range_frame(
                stamp=stamp,
                cycle_board_frames=cycle_board_frames,
            )

    def _build_query_frame(self, device_id: int) -> bytes:
        payload = bytes([0x5A, 0xA5, STP23L_QUERY_COMMAND, device_id & 0xFF])
        crc = crc16_modbus(payload)
        return payload + crc.to_bytes(2, byteorder="little")

    def _wait_for_response_active(
        self, handle: Any, expected_device_id: int
    ) -> Optional[BoardFrame]:
        deadline = time.monotonic() + self.serial_response_timeout_sec
        while not self._serial_stop_event.is_set():
            with self._parser_lock:
                board_frame = self._pending_board_frames.pop(expected_device_id, None)
                if board_frame is not None:
                    return board_frame
                if time.monotonic() >= deadline:
                    return None

            self._drain_serial_input(handle)

            with self._parser_lock:
                board_frame = self._pending_board_frames.pop(expected_device_id, None)
                if board_frame is not None:
                    return board_frame
                if time.monotonic() >= deadline:
                    return None

            time.sleep(0.0005)

        return None

    def _wait_for_infrared_response_active(
        self,
        handle: Any,
        expected_device_id: int,
        response_timeout_sec: float,
    ) -> Optional[InfraredFrame]:
        deadline = time.monotonic() + max(0.0, float(response_timeout_sec))
        while not self._serial_stop_event.is_set():
            with self._parser_lock:
                infrared_frame = self._pending_infrared_frames.pop(
                    expected_device_id, None
                )
                if infrared_frame is not None:
                    return infrared_frame
                if time.monotonic() >= deadline:
                    return None

            self._drain_serial_input(handle)

            with self._parser_lock:
                infrared_frame = self._pending_infrared_frames.pop(
                    expected_device_id, None
                )
                if infrared_frame is not None:
                    return infrared_frame
                if time.monotonic() >= deadline:
                    return None

            time.sleep(0.0005)

        return None

    def _snapshot_rx_debug_state(self) -> Tuple[int, List[int], List[int]]:
        with self._parser_lock:
            return (
                len(self._serial_byte_buffer),
                sorted(self._pending_board_frames.keys()),
                sorted(self._pending_infrared_frames.keys()),
            )

    def _clear_pending_query_responses(self) -> None:
        with self._parser_lock:
            for device_id in self.serial_query_device_ids:
                self._pending_board_frames.pop(device_id, None)

    def _clear_pending_infrared_query_responses(self) -> None:
        with self._parser_lock:
            self._pending_infrared_frames.clear()

    # ---- Buffer parser ------------------------------------------------------

    def _drain_serial_buffer_locked(self) -> bool:
        produced_frame = False
        while True:
            header_index = self._serial_byte_buffer.find(STP23L_HEADER)
            if header_index < 0:
                if len(self._serial_byte_buffer) > STP23L_DATA_FRAME_LEN:
                    del self._serial_byte_buffer[:-1]
                return produced_frame
            if header_index > 0:
                del self._serial_byte_buffer[:header_index]
            if len(self._serial_byte_buffer) < STP23L_QUERY_FRAME_LEN:
                return produced_frame

            command = self._serial_byte_buffer[2]
            if command == STP23L_QUERY_COMMAND:
                if len(self._serial_byte_buffer) < STP23L_QUERY_FRAME_LEN:
                    return produced_frame
                if self._is_valid_query_echo(bytes(self._serial_byte_buffer[:6])):
                    del self._serial_byte_buffer[:STP23L_QUERY_FRAME_LEN]
                else:
                    del self._serial_byte_buffer[:1]
                continue

            if command == INFRARED_QUERY_COMMAND:
                if len(self._serial_byte_buffer) < INFRARED_QUERY_FRAME_LEN:
                    return produced_frame
                if self._is_valid_infrared_query_echo(bytes(self._serial_byte_buffer[:6])):
                    del self._serial_byte_buffer[:INFRARED_QUERY_FRAME_LEN]
                else:
                    del self._serial_byte_buffer[:1]
                continue

            if command == INFRARED_DATA_COMMAND:
                if len(self._serial_byte_buffer) < INFRARED_DATA_FRAME_LEN:
                    return produced_frame
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
                self._record_infrared_frame_locked(
                    self._decode_infrared_frame(frame_bytes)
                )
                produced_frame = True
                continue

            if command != STP23L_DATA_COMMAND:
                del self._serial_byte_buffer[:1]
                continue

            if len(self._serial_byte_buffer) < STP23L_DATA_FRAME_LEN:
                return produced_frame

            frame_bytes = bytes(self._serial_byte_buffer[:STP23L_DATA_FRAME_LEN])
            payload_crc = int.from_bytes(frame_bytes[14:16], byteorder="little")
            expected_crc = crc16_modbus(frame_bytes[:14])
            if payload_crc != expected_crc:
                self._logger.warn(
                    "Dropped STP23L frame due to CRC mismatch "
                    f"(expected=0x{expected_crc:04X}, got=0x{payload_crc:04X})"
                )
                del self._serial_byte_buffer[:1]
                continue

            del self._serial_byte_buffer[:STP23L_DATA_FRAME_LEN]
            board_frame = self._decode_stp23l_frame(frame_bytes)
            self._record_board_frame_locked(board_frame)
            produced_frame = True

    def _is_valid_query_echo(self, frame_bytes: bytes) -> bool:
        if len(frame_bytes) != STP23L_QUERY_FRAME_LEN:
            return False
        payload_crc = int.from_bytes(frame_bytes[4:6], byteorder="little")
        expected_crc = crc16_modbus(frame_bytes[:4])
        return payload_crc == expected_crc

    def _is_valid_infrared_query_echo(self, frame_bytes: bytes) -> bool:
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

    def _record_infrared_frame_locked(self, infrared_frame: InfraredFrame) -> None:
        self._pending_infrared_frames[infrared_frame.device_id] = infrared_frame
        if self._infrared_layer is not None:
            self._infrared_layer.handle_infrared_frame(infrared_frame)

    def _record_board_frame_locked(self, board_frame: BoardFrame) -> None:
        self._pending_board_frames[board_frame.device_id] = board_frame
        self._latest_board_frames[board_frame.device_id] = board_frame
        self._ingest_board_frame(board_frame)
        self._log_decoded_board_frame_locked(board_frame)

    def _decode_stp23l_frame(self, frame_bytes: bytes) -> BoardFrame:
        rx_stamp = self._clock.now()
        distances_mm = [
            int.from_bytes(frame_bytes[6:8], byteorder="little"),
            int.from_bytes(frame_bytes[8:10], byteorder="little"),
            int.from_bytes(frame_bytes[10:12], byteorder="little"),
            int.from_bytes(frame_bytes[12:14], byteorder="little"),
        ]
        return BoardFrame(
            rx_stamp=rx_stamp,
            device_id=int(frame_bytes[3]),
            report_type=int(frame_bytes[4]),
            status_bits=int(frame_bytes[5]),
            distances_mm=distances_mm,
        )

    def _log_decoded_board_frame_locked(self, board_frame: BoardFrame) -> None:
        mappings = [
            mapping
            for mapping in self._sensor_map.values()
            if mapping.device_id == board_frame.device_id
        ]
        if not mappings:
            self._logger.warn(
                "Decoded STP23L frame from unmapped "
                f"device_id={board_frame.device_id} status=0x{board_frame.status_bits:02X}"
            )
            return

        usable_count = sum(
            1 for mapping in mappings if self._is_logical_sensor_usable(mapping, board_frame)
        )
        should_log_info = (
            self.serial_decode_log_enabled
            and self._should_emit_decode_log_locked(board_frame.device_id, board_frame.rx_stamp)
        )
        if not should_log_info and usable_count > 0:
            return

        message = self._format_decoded_board_frame_log(board_frame, mappings)
        if usable_count == 0:
            self._logger.warn(
                "Decoded STP23L frame but all mapped sensors are unusable: " + message
            )
            return
        self._logger.info("Decoded STP23L frame: " + message)

    def _should_emit_decode_log_locked(self, device_id: int, stamp: Time) -> bool:
        if self.serial_decode_log_interval_ms <= 0.0:
            self._last_decode_log_stamp_ns_by_device[device_id] = stamp.nanoseconds
            return True
        last_stamp_ns = self._last_decode_log_stamp_ns_by_device.get(device_id)
        if last_stamp_ns is None:
            self._last_decode_log_stamp_ns_by_device[device_id] = stamp.nanoseconds
            return True
        if (stamp.nanoseconds - last_stamp_ns) / 1e6 < self.serial_decode_log_interval_ms:
            return False
        self._last_decode_log_stamp_ns_by_device[device_id] = stamp.nanoseconds
        return True

    def _format_decoded_board_frame_log(
        self,
        board_frame: BoardFrame,
        mappings: List[SerialSensorMapping],
    ) -> str:
        slot_summary = ", ".join(
            f"s{slot_index + 1}={self._format_raw_mm(raw_mm)}/"
            f"{'on' if board_frame.status_bits & (1 << slot_index) else 'off'}"
            for slot_index, raw_mm in enumerate(board_frame.distances_mm)
        )
        mapped_summary = ", ".join(
            self._format_mapping_summary(mapping, board_frame) for mapping in mappings
        )
        logical_snapshot = ", ".join(
            f"{self._logical_sensor_short_name(name)}="
            f"{'Y' if self._latest_serial_valid[name] else 'N'}"
            for name in SENSOR_ORDER
        )
        report_type_name = self._report_type_name(board_frame.report_type)
        return (
            f"device_id={board_frame.device_id} "
            f"report={report_type_name} "
            f"status=0x{board_frame.status_bits:02X} "
            f"slots=[{slot_summary}] "
            f"mapped=[{mapped_summary}] "
            f"logical_valid=[{logical_snapshot}]"
        )

    def _format_mapping_summary(
        self,
        mapping: SerialSensorMapping,
        board_frame: BoardFrame,
    ) -> str:
        raw_mm = board_frame.distances_mm[mapping.slot_index]
        online = bool(board_frame.status_bits & (1 << mapping.slot_index))
        serial_valid = online and raw_mm != STP23L_INVALID_DISTANCE_MM
        usable = self._is_logical_sensor_usable(mapping, board_frame)
        return (
            f"{mapping.logical_sensor}@s{mapping.slot_index + 1}="
            f"{self._format_raw_mm(raw_mm)} "
            f"serial={'Y' if serial_valid else 'N'} "
            f"usable={'Y' if usable else 'N'}"
        )

    def _is_logical_sensor_usable(
        self,
        mapping: SerialSensorMapping,
        board_frame: BoardFrame,
    ) -> bool:
        raw_mm = board_frame.distances_mm[mapping.slot_index]
        online = bool(board_frame.status_bits & (1 << mapping.slot_index))
        if not online or raw_mm == STP23L_INVALID_DISTANCE_MM:
            return False
        mount = self._sensor_mounts[mapping.logical_sensor]
        value_m = float(raw_mm) / 1000.0
        return math.isfinite(value_m) and mount.min_range_m <= value_m <= mount.max_range_m

    def _format_raw_mm(self, raw_mm: int) -> str:
        if raw_mm == STP23L_INVALID_DISTANCE_MM:
            return "FFFF"
        return str(raw_mm)

    def _logical_sensor_short_name(self, sensor_name: str) -> str:
        abbreviations = {
            "front_center": "FC",
            "rear_center": "RC",
            "left_front": "LF",
            "left_rear": "LR",
            "right_front": "RF",
            "right_rear": "RR",
        }
        return abbreviations.get(sensor_name, sensor_name)

    def _report_type_name(self, report_type: int) -> str:
        if report_type == 0x00:
            return "active"
        if report_type == 0x01:
            return "query"
        return f"0x{report_type:02X}"

    def _build_device_status_snapshot(
        self,
        *,
        board_frame: BoardFrame,
        now: Time,
    ) -> Dict[str, Any]:
        slots = []
        for slot_index, raw_mm in enumerate(board_frame.distances_mm):
            online = bool(board_frame.status_bits & (1 << slot_index))
            serial_valid = online and raw_mm != STP23L_INVALID_DISTANCE_MM
            slots.append(
                {
                    "slot_index": slot_index,
                    "raw_mm": raw_mm,
                    "online": online,
                    "serial_valid": serial_valid,
                    "range_m": (
                        float(raw_mm) / 1000.0 if serial_valid else None
                    ),
                }
            )
        return {
            "device_id": board_frame.device_id,
            "age_ms": abs_time_diff_ms(now, board_frame.rx_stamp),
            "report_type": board_frame.report_type,
            "report_name": self._report_type_name(board_frame.report_type),
            "status_bits": board_frame.status_bits,
            "slots": slots,
        }

    def _build_logical_sensor_status_snapshot(
        self,
        sensor_name: str,
    ) -> Dict[str, Any]:
        mapping = self._sensor_map[sensor_name]
        board_frame = self._latest_board_frames.get(mapping.device_id)
        raw_mm: Optional[int] = None
        online: Optional[bool] = None
        serial_valid = False
        usable = False
        if board_frame is not None:
            raw_mm = board_frame.distances_mm[mapping.slot_index]
            online = bool(board_frame.status_bits & (1 << mapping.slot_index))
            serial_valid = online and raw_mm != STP23L_INVALID_DISTANCE_MM
            usable = self._is_logical_sensor_usable(mapping, board_frame)

        range_m = self._latest_serial_ranges_m[sensor_name]
        return {
            "device_id": mapping.device_id,
            "slot_index": mapping.slot_index,
            "raw_mm": raw_mm,
            "online": online,
            "serial_valid": serial_valid,
            "range_m": float(range_m) if math.isfinite(range_m) else None,
            "usable": usable,
        }

    def _ingest_board_frame(self, board_frame: BoardFrame) -> None:
        for mapping in self._sensor_map.values():
            if mapping.device_id != board_frame.device_id:
                continue
            slot_index = mapping.slot_index
            raw_mm = board_frame.distances_mm[slot_index]
            online = bool(board_frame.status_bits & (1 << slot_index))
            self._latest_serial_ranges_m[mapping.logical_sensor] = float(raw_mm) / 1000.0
            self._latest_serial_valid[mapping.logical_sensor] = (
                online and raw_mm != STP23L_INVALID_DISTANCE_MM
            )

    def _maybe_publish_cycle_range_frame(
        self,
        *,
        stamp: Time,
        cycle_board_frames: Dict[int, BoardFrame],
    ) -> None:
        if not self._should_publish_serial_frame(stamp):
            return
        frame = self._build_range_frame_from_cycle(stamp, cycle_board_frames)
        with self._latest_range_frame_lock:
            self._latest_range_frame = frame
        self._last_serial_frame_stamp = stamp

    def _clear_latest_range_state(self) -> None:
        with self._latest_range_frame_lock:
            self._latest_range_frame = None
        self._last_serial_frame_stamp = None
        with self._parser_lock:
            self._latest_serial_ranges_m = {
                name: float("nan") for name in SENSOR_ORDER
            }
            self._latest_serial_valid = {
                name: False for name in SENSOR_ORDER
            }

    def _should_publish_serial_frame(self, stamp: Time) -> bool:
        if self._last_serial_frame_stamp is None:
            return True
        min_interval = Duration(
            nanoseconds=int(self.serial_min_publish_interval_ms * 1e6)
        )
        return (stamp - self._last_serial_frame_stamp) >= min_interval

    def _build_range_frame_from_cycle(
        self,
        stamp: Time,
        cycle_board_frames: Dict[int, BoardFrame],
    ) -> RangeFrame:
        ranges: Dict[str, float] = {}
        valid: Dict[str, bool] = {}
        for name in SENSOR_ORDER:
            mapping = self._sensor_map[name]
            mount = self._sensor_mounts[name]
            board_frame = cycle_board_frames.get(mapping.device_id)
            if board_frame is None:
                ranges[name] = float("nan")
                valid[name] = False
                continue
            raw_mm = board_frame.distances_mm[mapping.slot_index]
            value = float(raw_mm) / 1000.0
            online = bool(board_frame.status_bits & (1 << mapping.slot_index))
            serial_valid = online and raw_mm != STP23L_INVALID_DISTANCE_MM
            ranges[name] = value
            range_valid = (
                math.isfinite(value)
                and mount.min_range_m <= value <= mount.max_range_m
            )
            valid[name] = serial_valid and range_valid
        return RangeFrame(stamp=stamp, ranges=ranges, valid=valid)
