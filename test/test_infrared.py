from __future__ import annotations

import json
import unittest

from _module_stubs import install_test_stubs

install_test_stubs()

from rclpy.time import Time

from agv_pose_refiner_py.infrared import (
    InfraredConfig,
    InfraredEventProcessor,
    InfraredFrame,
    InfraredRule,
    parse_infrared_config,
)
from agv_pose_refiner_py.infrared_receiver import InfraredReceiveLayer


class DummyLogger:
    def __init__(self) -> None:
        self.messages = []

    def info(self, msg: str) -> None:
        self.messages.append(("info", msg))

    def warn(self, msg: str) -> None:
        self.messages.append(("warn", msg))

    def error(self, msg: str) -> None:
        self.messages.append(("error", msg))


class DummyClock:
    def __init__(self) -> None:
        self.now_time = Time(nanoseconds=0)

    def now(self) -> Time:
        return self.now_time


class DummyPublisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, msg: object) -> None:
        self.messages.append(msg)


class DummyNode:
    def __init__(self) -> None:
        self.publishers = {}
        self.logger = DummyLogger()
        self.clock = DummyClock()

    def create_publisher(
        self, _msg_type: type, topic: str, _queue_size: int
    ) -> DummyPublisher:
        publisher = DummyPublisher()
        self.publishers[topic] = publisher
        return publisher

    def get_logger(self) -> DummyLogger:
        return self.logger

    def get_clock(self) -> DummyClock:
        return self.clock


class TestInfraredConfig(unittest.TestCase):
    def test_parse_infrared_config(self) -> None:
        config = parse_infrared_config(
            {
                "scene_manager": {"active_scene": "mode_red"},
                "infrared": {
                    "use_topic": "/infrared",
                    "debug_topic": "/infrared_debug",
                    "scenes": {
                        "mode_red": {
                            "rules": [
                                {
                                    "x_range": [0.0, 2.0],
                                    "raw_bytes": [0x11, "0x22"],
                                    "mapped_type": "test",
                                    "send_to_topic": "0x01",
                                }
                            ]
                        }
                    },
                },
            }
        )
        self.assertEqual(config.active_scene, "mode_red")
        self.assertEqual(config.use_topic, "/infrared")
        self.assertEqual(config.debug_topic, "/infrared_debug")
        self.assertEqual(config.scenes["mode_red"][0].raw_bytes, (0x11, 0x22))
        self.assertEqual(config.scenes["mode_red"][0].send_to_topic, 0x01)


class TestInfraredEventProcessor(unittest.TestCase):
    def setUp(self) -> None:
        self.latest_coarse = (1.0, Time(nanoseconds=1_000_000_000))
        self.processor = InfraredEventProcessor(
            config=InfraredConfig(
                active_scene="mode_red",
                use_topic="/infrared",
                debug_topic="/infrared_debug",
                max_coarse_pose_age_ms=500.0,
                scenes={
                    "mode_red": (
                        InfraredRule(
                            x_min=0.0,
                            x_max=2.0,
                            raw_bytes=(0x11, 0x22, 0x33),
                            mapped_type="spear_done_continue",
                            send_to_topic=0x01,
                        ),
                        InfraredRule(
                            x_min=0.0,
                            x_max=2.0,
                            raw_bytes=(0x01, 0x02, 0x03),
                            mapped_type="spear_done_wait",
                            send_to_topic=0x02,
                        ),
                    )
                },
            ),
            latest_coarse_x_provider=lambda: self.latest_coarse,
        )

    def _frame(
        self,
        *,
        rx_ms: int,
        device_id: int,
        raw_byte: int,
        device_ts_ms: int,
    ) -> InfraredFrame:
        return InfraredFrame(
            rx_stamp=Time(nanoseconds=rx_ms * 1_000_000),
            device_id=device_id,
            report_type=0x01,
            raw_byte=raw_byte,
            device_timestamp_ms=device_ts_ms,
        )

    def test_first_frame_only_syncs(self) -> None:
        result = self.processor.process_frame(
            self._frame(rx_ms=1000, device_id=3, raw_byte=0x11, device_ts_ms=120)
        )
        self.assertEqual(result.action, "synced")
        self.assertEqual(result.reason, "SYNC_ESTABLISHED")
        self.assertIsNone(self.processor.snapshot_shared_last_event())

    def test_publishes_only_when_raw_byte_changes_and_timestamp_increases(self) -> None:
        self.processor.process_frame(
            self._frame(rx_ms=990, device_id=4, raw_byte=0x11, device_ts_ms=980)
        )
        self.processor.process_frame(
            self._frame(rx_ms=1000, device_id=3, raw_byte=0x11, device_ts_ms=120)
        )
        published = self.processor.process_frame(
            self._frame(rx_ms=1010, device_id=3, raw_byte=0x11, device_ts_ms=130)
        )
        self.assertEqual(published.action, "published")
        self.assertEqual(published.event.mapped_byte, 0x01)

        duplicate_byte = self.processor.process_frame(
            self._frame(rx_ms=1030, device_id=4, raw_byte=0x11, device_ts_ms=1010)
        )
        self.assertEqual(duplicate_byte.action, "dropped")
        self.assertEqual(duplicate_byte.reason, "RAW_BYTE_DUPLICATED")

    def test_same_mapped_topic_republishes_when_raw_byte_changes(self) -> None:
        self.processor.process_frame(
            self._frame(rx_ms=1000, device_id=3, raw_byte=0x11, device_ts_ms=120)
        )
        first = self.processor.process_frame(
            self._frame(rx_ms=1010, device_id=3, raw_byte=0x11, device_ts_ms=130)
        )
        self.assertEqual(first.action, "published")
        self.assertEqual(first.event.mapped_byte, 0x01)

        second = self.processor.process_frame(
            self._frame(rx_ms=1020, device_id=3, raw_byte=0x22, device_ts_ms=140)
        )
        self.assertEqual(second.action, "published")
        self.assertEqual(second.event.mapped_byte, 0x01)

        older_ts = self.processor.process_frame(
            self._frame(rx_ms=1040, device_id=3, raw_byte=0x22, device_ts_ms=130)
        )
        self.assertEqual(older_ts.action, "dropped")
        self.assertEqual(older_ts.reason, "DEVICE_TIMESTAMP_ROLLBACK")

    def test_zero_byte_is_dropped_before_mapping(self) -> None:
        self.processor.process_frame(
            self._frame(rx_ms=1000, device_id=3, raw_byte=0x11, device_ts_ms=120)
        )
        result = self.processor.process_frame(
            self._frame(rx_ms=1010, device_id=3, raw_byte=0x00, device_ts_ms=130)
        )
        self.assertEqual(result.action, "dropped")
        self.assertEqual(result.reason, "RAW_BYTE_ZERO")

    def test_no_coarse_x_does_not_poison_future_publish(self) -> None:
        self.latest_coarse = None
        self.processor.process_frame(
            self._frame(rx_ms=1000, device_id=3, raw_byte=0x11, device_ts_ms=120)
        )
        result = self.processor.process_frame(
            self._frame(rx_ms=1010, device_id=3, raw_byte=0x22, device_ts_ms=130)
        )
        self.assertEqual(result.action, "dropped")
        self.assertEqual(result.reason, "NO_COARSE_X")
        self.assertIsNone(self.processor.snapshot_shared_last_event())

        self.latest_coarse = (1.0, Time(nanoseconds=1_000_000_000))
        published = self.processor.process_frame(
            self._frame(rx_ms=1020, device_id=3, raw_byte=0x22, device_ts_ms=140)
        )
        self.assertEqual(published.action, "published")
        self.assertEqual(published.event.mapped_byte, 0x01)

    def test_stale_coarse_x_is_dropped(self) -> None:
        self.latest_coarse = (1.0, Time(nanoseconds=100_000_000))
        self.processor.process_frame(
            self._frame(rx_ms=1000, device_id=3, raw_byte=0x11, device_ts_ms=120)
        )
        result = self.processor.process_frame(
            self._frame(rx_ms=1010, device_id=3, raw_byte=0x22, device_ts_ms=130)
        )
        self.assertEqual(result.action, "dropped")
        self.assertEqual(result.reason, "COARSE_X_TOO_OLD")
        self.assertIsNone(self.processor.snapshot_shared_last_event())

    def test_same_raw_byte_is_deduplicated_even_if_mapping_differs(self) -> None:
        self.processor = InfraredEventProcessor(
            config=InfraredConfig(
                active_scene="mode_red",
                use_topic="/infrared",
                debug_topic="/infrared_debug",
                max_coarse_pose_age_ms=500.0,
                scenes={
                    "mode_red": (
                        InfraredRule(
                            x_min=0.0,
                            x_max=2.0,
                            raw_bytes=(0x11,),
                            mapped_type="near",
                            send_to_topic=0x01,
                        ),
                        InfraredRule(
                            x_min=9.0,
                            x_max=12.0,
                            raw_bytes=(0x11,),
                            mapped_type="far",
                            send_to_topic=0x03,
                        ),
                    )
                },
            ),
            latest_coarse_x_provider=lambda: self.latest_coarse,
        )
        self.processor.process_frame(
            self._frame(rx_ms=1000, device_id=3, raw_byte=0x00, device_ts_ms=120)
        )
        self.processor.process_frame(
            self._frame(rx_ms=1005, device_id=4, raw_byte=0x00, device_ts_ms=1000)
        )

        self.latest_coarse = (1.0, Time(nanoseconds=1_000_000_000))
        near_event = self.processor.process_frame(
            self._frame(rx_ms=1010, device_id=3, raw_byte=0x11, device_ts_ms=130)
        )
        self.assertEqual(near_event.action, "published")
        self.assertEqual(near_event.event.mapped_byte, 0x01)

        self.latest_coarse = (10.0, Time(nanoseconds=1_020_000_000))
        far_event = self.processor.process_frame(
            self._frame(rx_ms=1030, device_id=4, raw_byte=0x11, device_ts_ms=1020)
        )
        self.assertEqual(far_event.action, "dropped")
        self.assertEqual(far_event.reason, "RAW_BYTE_DUPLICATED")

    def test_snapshot_shared_last_event_exposes_raw_byte(self) -> None:
        self.processor.process_frame(
            self._frame(rx_ms=1000, device_id=3, raw_byte=0x11, device_ts_ms=120)
        )
        published = self.processor.process_frame(
            self._frame(rx_ms=1010, device_id=3, raw_byte=0x22, device_ts_ms=130)
        )
        self.assertEqual(published.action, "published")

        shared_event = self.processor.snapshot_shared_last_event()
        self.assertIsNotNone(shared_event)
        self.assertEqual(shared_event.raw_byte, 0x22)
        self.assertEqual(shared_event.mapped_byte, 0x01)

    def test_device_timestamp_rollback_requires_resync(self) -> None:
        self.processor.process_frame(
            self._frame(rx_ms=1000, device_id=3, raw_byte=0x11, device_ts_ms=120)
        )
        self.processor.process_frame(
            self._frame(rx_ms=1010, device_id=3, raw_byte=0x22, device_ts_ms=130)
        )
        rollback = self.processor.process_frame(
            self._frame(rx_ms=1020, device_id=3, raw_byte=0x33, device_ts_ms=80)
        )
        self.assertEqual(rollback.action, "dropped")
        self.assertEqual(rollback.reason, "DEVICE_TIMESTAMP_ROLLBACK")

        resync = self.processor.process_frame(
            self._frame(rx_ms=1030, device_id=3, raw_byte=0x33, device_ts_ms=81)
        )
        self.assertEqual(resync.action, "synced")
        self.assertEqual(resync.reason, "SYNC_ESTABLISHED")


class TestInfraredReceiveLayer(unittest.TestCase):
    def test_publish_event_emits_topic_and_debug_json(self) -> None:
        node = DummyNode()
        layer = InfraredReceiveLayer(
            node=node,
            config=InfraredConfig(
                active_scene="mode_red",
                use_topic="/infrared",
                debug_topic="/infrared_debug",
                max_coarse_pose_age_ms=500.0,
                scenes={
                    "mode_red": (
                        InfraredRule(
                            x_min=0.0,
                            x_max=2.0,
                            raw_bytes=(0x11,),
                            mapped_type="spear_done_continue",
                            send_to_topic=0x01,
                        ),
                    )
                },
            ),
            query_device_ids=[3, 4],
            latest_coarse_x_provider=lambda: (1.0, Time(nanoseconds=1_000_000_000)),
            serial_port="/dev/infrared_serial",
            serial_baudrate=115200,
            serial_response_timeout_sec=0.02,
            serial_poll_rate_hz=100.0,
        )

        layer._handle_frame_locked(
            InfraredFrame(
                rx_stamp=Time(nanoseconds=1_000_000_000),
                device_id=3,
                report_type=0x01,
                raw_byte=0x11,
                device_timestamp_ms=100,
            )
        )
        layer._handle_frame_locked(
            InfraredFrame(
                rx_stamp=Time(nanoseconds=1_010_000_000),
                device_id=3,
                report_type=0x01,
                raw_byte=0x11,
                device_timestamp_ms=110,
            )
        )

        topic_pub = node.publishers["/infrared"]
        debug_pub = node.publishers["/infrared_debug"]
        self.assertEqual(len(topic_pub.messages), 1)
        self.assertEqual(topic_pub.messages[0].data, 0x01)
        self.assertEqual(len(debug_pub.messages), 1)
        debug_payload = json.loads(debug_pub.messages[0].data)
        self.assertEqual(debug_payload["mapped_type"], "spear_done_continue")
        self.assertEqual(debug_payload["device_id"], 3)
        self.assertEqual(debug_payload["raw_byte"], 0x11)
        self.assertEqual(debug_payload["mapped_byte"], 0x01)
        self.assertEqual(debug_payload["scene"], "mode_red")
        self.assertAlmostEqual(debug_payload["x"], 1.0, places=6)

    def test_snapshot_status_includes_shared_raw_byte(self) -> None:
        node = DummyNode()
        layer = InfraredReceiveLayer(
            node=node,
            config=InfraredConfig(
                active_scene="mode_red",
                use_topic="/infrared",
                debug_topic="/infrared_debug",
                max_coarse_pose_age_ms=500.0,
                scenes={
                    "mode_red": (
                        InfraredRule(
                            x_min=0.0,
                            x_max=2.0,
                            raw_bytes=(0x11,),
                            mapped_type="spear_done_continue",
                            send_to_topic=0x01,
                        ),
                    )
                },
            ),
            query_device_ids=[3, 4],
            latest_coarse_x_provider=lambda: (1.0, Time(nanoseconds=1_000_000_000)),
            serial_port="/dev/infrared_serial",
            serial_baudrate=115200,
            serial_response_timeout_sec=0.02,
            serial_poll_rate_hz=100.0,
        )

        layer._handle_frame_locked(
            InfraredFrame(
                rx_stamp=Time(nanoseconds=1_000_000_000),
                device_id=3,
                report_type=0x01,
                raw_byte=0x11,
                device_timestamp_ms=100,
            )
        )
        layer._handle_frame_locked(
            InfraredFrame(
                rx_stamp=Time(nanoseconds=1_010_000_000),
                device_id=3,
                report_type=0x01,
                raw_byte=0x11,
                device_timestamp_ms=110,
            )
        )

        snapshot = layer.snapshot_status()
        self.assertEqual(snapshot["shared_last_event"]["raw_byte"], 0x11)
        self.assertEqual(snapshot["shared_last_event"]["source_device_id"], 3)
        self.assertEqual(snapshot["shared_last_event"]["aligned_ts_ms"], 1010)


if __name__ == "__main__":
    unittest.main()
