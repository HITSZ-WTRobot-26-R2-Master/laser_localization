from __future__ import annotations

import json
import time
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


class DummyHandle:
    def __init__(self) -> None:
        self.writes = []
        self.flush_count = 0

    def write(self, payload: bytes) -> None:
        self.writes.append(payload)

    def flush(self) -> None:
        self.flush_count += 1


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

    def test_parse_infrared_config_accepts_runtime_topic_overrides(self) -> None:
        config = parse_infrared_config(
            {
                "scene_manager": {"active_scene": "mode_red"},
                "infrared": {
                    "use_topic": "/infrared_legacy",
                    "debug_topic": "/infrared_debug_legacy",
                    "scenes": {
                        "mode_red": {
                            "rules": [
                                {
                                    "x_range": [0.0, 2.0],
                                    "raw_bytes": [0x11],
                                    "mapped_type": "test",
                                    "send_to_topic": "0x01",
                                }
                            ]
                        }
                    },
                },
            },
            runtime_config={
                "use_topic": "/infrared",
                "debug_topic": "/infrared_debug",
            },
        )
        self.assertEqual(config.use_topic, "/infrared")
        self.assertEqual(config.debug_topic, "/infrared_debug")

    def test_parse_infrared_config_falls_back_when_runtime_topics_empty(self) -> None:
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
                                    "raw_bytes": [0x11],
                                    "mapped_type": "test",
                                    "send_to_topic": "0x01",
                                }
                            ]
                        }
                    },
                },
            },
            runtime_config={
                "use_topic": "",
                "debug_topic": " ",
            },
        )
        self.assertEqual(config.use_topic, "/infrared")
        self.assertEqual(config.debug_topic, "/infrared_debug")

    def test_parse_infrared_config_uses_defaults_when_topics_missing(self) -> None:
        config = parse_infrared_config(
            {
                "scene_manager": {"active_scene": "mode_red"},
                "infrared": {
                    "scenes": {
                        "mode_red": {
                            "rules": [
                                {
                                    "x_range": [0.0, 2.0],
                                    "raw_bytes": [0x11],
                                    "mapped_type": "test",
                                    "send_to_topic": "0x01",
                                }
                            ]
                        }
                    },
                },
            }
        )
        self.assertEqual(config.use_topic, "/infrared")
        self.assertEqual(config.debug_topic, "/infrared_debug")


class TestInfraredEventProcessor(unittest.TestCase):
    def setUp(self) -> None:
        self.latest_coarse = (1.0, Time(nanoseconds=1_000_000_000))
        self.processor = InfraredEventProcessor(
            config=InfraredConfig(
                active_scene="mode_red",
                use_topic="/infrared",
                debug_topic="/infrared_debug",
                raw_topic="/infrared_raw",
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

    def _new_processor(
        self,
        rules: tuple[InfraredRule, ...],
    ) -> InfraredEventProcessor:
        return InfraredEventProcessor(
            config=InfraredConfig(
                active_scene="mode_red",
                use_topic="/infrared",
                debug_topic="/infrared_debug",
                raw_topic="/infrared_raw",
                max_coarse_pose_age_ms=500.0,
                scenes={"mode_red": rules},
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

    def test_first_valid_value_only_observes_without_publish(self) -> None:
        self.processor.process_frame(
            self._frame(rx_ms=990, device_id=4, raw_byte=0x11, device_ts_ms=980)
        )
        self.processor.process_frame(
            self._frame(rx_ms=1000, device_id=3, raw_byte=0x11, device_ts_ms=120)
        )
        observed = self.processor.process_frame(
            self._frame(rx_ms=1010, device_id=3, raw_byte=0x11, device_ts_ms=130)
        )
        self.assertEqual(observed.action, "dropped")
        self.assertEqual(observed.reason, "STARTUP_FIRST_VALID_OBSERVED")
        self.assertIsNone(self.processor.snapshot_shared_last_event())

        duplicate_byte = self.processor.process_frame(
            self._frame(rx_ms=1030, device_id=4, raw_byte=0x11, device_ts_ms=1010)
        )
        self.assertEqual(duplicate_byte.action, "dropped")
        self.assertEqual(duplicate_byte.reason, "WAITING_FOR_FIRST_TRANSITION")

    def test_first_transition_sets_baseline_then_next_change_publishes(self) -> None:
        self.processor.process_frame(
            self._frame(rx_ms=1000, device_id=3, raw_byte=0x11, device_ts_ms=120)
        )
        observed = self.processor.process_frame(
            self._frame(rx_ms=1010, device_id=3, raw_byte=0x11, device_ts_ms=130)
        )
        self.assertEqual(observed.reason, "STARTUP_FIRST_VALID_OBSERVED")

        baseline = self.processor.process_frame(
            self._frame(rx_ms=1020, device_id=3, raw_byte=0x22, device_ts_ms=140)
        )
        self.assertEqual(baseline.action, "dropped")
        self.assertEqual(baseline.reason, "FIRST_TRANSITION_SET_AS_BASELINE")

        first_publish = self.processor.process_frame(
            self._frame(rx_ms=1030, device_id=3, raw_byte=0x33, device_ts_ms=150)
        )
        self.assertEqual(first_publish.action, "published")
        self.assertEqual(first_publish.event.mapped_byte, 0x01)

        second = self.processor.process_frame(
            self._frame(rx_ms=1040, device_id=3, raw_byte=0x11, device_ts_ms=160)
        )
        self.assertEqual(second.action, "published")
        self.assertEqual(second.event.mapped_byte, 0x01)

        older_ts = self.processor.process_frame(
            self._frame(rx_ms=1050, device_id=3, raw_byte=0x22, device_ts_ms=130)
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
        observed = self.processor.process_frame(
            self._frame(rx_ms=1020, device_id=3, raw_byte=0x22, device_ts_ms=140)
        )
        self.assertEqual(observed.action, "dropped")
        self.assertEqual(observed.reason, "STARTUP_FIRST_VALID_OBSERVED")

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

    def test_x_outside_active_window_clears_raw_byte_memory_without_publish(self) -> None:
        self.processor.process_frame(
            self._frame(rx_ms=1000, device_id=3, raw_byte=0x11, device_ts_ms=120)
        )
        observed = self.processor.process_frame(
            self._frame(rx_ms=1010, device_id=3, raw_byte=0x11, device_ts_ms=130)
        )
        self.assertEqual(observed.action, "dropped")
        self.assertEqual(observed.reason, "STARTUP_FIRST_VALID_OBSERVED")
        self.assertEqual(self.processor._shared_last_raw_byte, 0x00)

        self.latest_coarse = (5.0, Time(nanoseconds=1_020_000_000))
        outside = self.processor.process_frame(
            self._frame(rx_ms=1020, device_id=3, raw_byte=0x22, device_ts_ms=140)
        )
        self.assertEqual(outside.action, "dropped")
        self.assertEqual(outside.reason, "OUTSIDE_ACTIVE_X_WINDOW")
        self.assertEqual(self.processor._shared_last_raw_byte, 0x00)
        shared_event = self.processor.snapshot_shared_last_event()
        self.assertIsNone(shared_event)

    def test_same_raw_byte_republishes_after_passing_through_inactive_x_window(self) -> None:
        self.processor = self._new_processor(
            (
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
                    raw_bytes=(0x11, 0x22),
                    mapped_type="far",
                    send_to_topic=0x03,
                ),
            )
        )

        self.processor.process_frame(
            self._frame(rx_ms=1000, device_id=3, raw_byte=0x11, device_ts_ms=120)
        )
        first = self.processor.process_frame(
            self._frame(rx_ms=1010, device_id=3, raw_byte=0x11, device_ts_ms=130)
        )
        self.assertEqual(first.action, "dropped")
        self.assertEqual(first.reason, "STARTUP_FIRST_VALID_OBSERVED")

        self.latest_coarse = (5.0, Time(nanoseconds=1_020_000_000))
        outside = self.processor.process_frame(
            self._frame(rx_ms=1020, device_id=3, raw_byte=0x22, device_ts_ms=140)
        )
        self.assertEqual(outside.reason, "OUTSIDE_ACTIVE_X_WINDOW")
        self.assertEqual(self.processor._shared_last_raw_byte, 0x00)

        self.latest_coarse = (10.0, Time(nanoseconds=1_030_000_000))
        second = self.processor.process_frame(
            self._frame(rx_ms=1030, device_id=3, raw_byte=0x11, device_ts_ms=150)
        )
        self.assertEqual(second.action, "dropped")
        self.assertEqual(second.reason, "STARTUP_FIRST_VALID_OBSERVED")

        third = self.processor.process_frame(
            self._frame(rx_ms=1040, device_id=3, raw_byte=0x22, device_ts_ms=160)
        )
        self.assertEqual(third.action, "dropped")
        self.assertEqual(third.reason, "FIRST_TRANSITION_SET_AS_BASELINE")

        published = self.processor.process_frame(
            self._frame(rx_ms=1050, device_id=3, raw_byte=0x11, device_ts_ms=170)
        )
        self.assertEqual(published.action, "published")
        self.assertEqual(published.event.mapped_byte, 0x03)

        published = self.processor.process_frame(
            self._frame(rx_ms=1060, device_id=3, raw_byte=0x33, device_ts_ms=180)
        )
        self.assertEqual(published.action, "dropped")
        self.assertEqual(published.reason, "NO_RULE_MATCH")

    def test_reenter_window_then_transition_publishes(self) -> None:
        self.processor = self._new_processor(
            (
                InfraredRule(
                    x_min=0.0,
                    x_max=2.0,
                    raw_bytes=(0x11, 0x22),
                    mapped_type="near",
                    send_to_topic=0x01,
                ),
                InfraredRule(
                    x_min=9.0,
                    x_max=12.0,
                    raw_bytes=(0x11, 0x22),
                    mapped_type="far",
                    send_to_topic=0x03,
                ),
            )
        )

        self.processor.process_frame(
            self._frame(rx_ms=1000, device_id=3, raw_byte=0x11, device_ts_ms=120)
        )
        self.processor.process_frame(
            self._frame(rx_ms=1010, device_id=3, raw_byte=0x11, device_ts_ms=130)
        )

        self.latest_coarse = (5.0, Time(nanoseconds=1_020_000_000))
        self.processor.process_frame(
            self._frame(rx_ms=1020, device_id=3, raw_byte=0x22, device_ts_ms=140)
        )

        self.latest_coarse = (10.0, Time(nanoseconds=1_030_000_000))
        reentered = self.processor.process_frame(
            self._frame(rx_ms=1030, device_id=3, raw_byte=0x11, device_ts_ms=150)
        )
        self.assertEqual(reentered.action, "dropped")
        self.assertEqual(reentered.reason, "STARTUP_FIRST_VALID_OBSERVED")

        baseline = self.processor.process_frame(
            self._frame(rx_ms=1040, device_id=3, raw_byte=0x22, device_ts_ms=160)
        )
        self.assertEqual(baseline.action, "dropped")
        self.assertEqual(baseline.reason, "FIRST_TRANSITION_SET_AS_BASELINE")

        published = self.processor.process_frame(
            self._frame(rx_ms=1050, device_id=3, raw_byte=0x11, device_ts_ms=170)
        )
        self.assertEqual(published.action, "published")
        self.assertEqual(published.event.mapped_byte, 0x03)

    def test_same_raw_byte_is_deduplicated_even_if_mapping_differs(self) -> None:
        self.processor = self._new_processor(
            (
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
        self.assertEqual(near_event.action, "dropped")
        self.assertEqual(near_event.reason, "STARTUP_FIRST_VALID_OBSERVED")

        self.latest_coarse = (10.0, Time(nanoseconds=1_020_000_000))
        far_event = self.processor.process_frame(
            self._frame(rx_ms=1030, device_id=4, raw_byte=0x11, device_ts_ms=1020)
        )
        self.assertEqual(far_event.action, "dropped")
        self.assertEqual(far_event.reason, "WAITING_FOR_FIRST_TRANSITION")

    def test_snapshot_shared_last_event_exposes_raw_byte(self) -> None:
        self.processor.process_frame(
            self._frame(rx_ms=1000, device_id=3, raw_byte=0x11, device_ts_ms=120)
        )
        observed = self.processor.process_frame(
            self._frame(rx_ms=1010, device_id=3, raw_byte=0x22, device_ts_ms=130)
        )
        self.assertEqual(observed.action, "dropped")
        self.assertEqual(observed.reason, "STARTUP_FIRST_VALID_OBSERVED")

        baseline = self.processor.process_frame(
            self._frame(rx_ms=1020, device_id=3, raw_byte=0x33, device_ts_ms=140)
        )
        self.assertEqual(baseline.action, "dropped")
        self.assertEqual(baseline.reason, "FIRST_TRANSITION_SET_AS_BASELINE")

        published = self.processor.process_frame(
            self._frame(rx_ms=1030, device_id=3, raw_byte=0x11, device_ts_ms=150)
        )
        self.assertEqual(published.action, "published")

        shared_event = self.processor.snapshot_shared_last_event()
        self.assertIsNotNone(shared_event)
        self.assertEqual(shared_event.raw_byte, 0x11)
        self.assertEqual(shared_event.mapped_byte, 0x01)

    def test_first_transition_baseline_returns_debug_event(self) -> None:
        self.processor.process_frame(
            self._frame(rx_ms=1000, device_id=3, raw_byte=0x11, device_ts_ms=120)
        )
        self.processor.process_frame(
            self._frame(rx_ms=1010, device_id=3, raw_byte=0x11, device_ts_ms=130)
        )

        baseline = self.processor.process_frame(
            self._frame(rx_ms=1020, device_id=3, raw_byte=0x22, device_ts_ms=140)
        )
        self.assertEqual(baseline.action, "dropped")
        self.assertEqual(baseline.reason, "FIRST_TRANSITION_SET_AS_BASELINE")
        self.assertIsNotNone(baseline.debug_event)
        self.assertEqual(baseline.debug_event.raw_byte, 0x22)
        self.assertEqual(baseline.debug_event.mapped_byte, 0x01)
        self.assertIsNone(baseline.event)

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
    def _build_layer(self, node: DummyNode) -> InfraredReceiveLayer:
        return InfraredReceiveLayer(
            node=node,
            config=InfraredConfig(
                active_scene="mode_red",
                use_topic="/infrared",
                debug_topic="/infrared_debug",
                raw_topic="/infrared_raw",
                max_coarse_pose_age_ms=500.0,
                scenes={
                    "mode_red": (
                        InfraredRule(
                            x_min=0.0,
                            x_max=2.0,
                            raw_bytes=(0x11, 0x22),
                            mapped_type="spear_done_continue",
                            send_to_topic=0x01,
                        ),
                    )
                },
            ),
            query_device_ids=[3, 4],
            latest_coarse_x_provider=lambda: (1.0, Time(nanoseconds=1_000_000_000)),
            serial_port="/dev/laser_serial",
            serial_baudrate=115200,
            serial_response_timeout_sec=0.02,
            serial_poll_rate_hz=100.0,
        )

    def test_publish_event_emits_topic_and_debug_json(self) -> None:
        node = DummyNode()
        layer = self._build_layer(node)

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
        layer._handle_frame_locked(
            InfraredFrame(
                rx_stamp=Time(nanoseconds=1_020_000_000),
                device_id=3,
                report_type=0x01,
                raw_byte=0x22,
                device_timestamp_ms=120,
            )
        )
        layer._handle_frame_locked(
            InfraredFrame(
                rx_stamp=Time(nanoseconds=1_030_000_000),
                device_id=3,
                report_type=0x01,
                raw_byte=0x11,
                device_timestamp_ms=130,
            )
        )

        topic_pub = node.publishers["/infrared"]
        debug_pub = node.publishers["/infrared_debug"]
        self.assertEqual(len(topic_pub.messages), 1)
        self.assertEqual(topic_pub.messages[0].data, 0x01)
        self.assertEqual(len(debug_pub.messages), 2)

        baseline_debug = json.loads(debug_pub.messages[0].data)
        self.assertEqual(baseline_debug["mapped_type"], "spear_done_continue")
        self.assertEqual(baseline_debug["device_id"], 3)
        self.assertEqual(baseline_debug["raw_byte"], 0x22)
        self.assertEqual(baseline_debug["mapped_byte"], 0x01)
        self.assertEqual(baseline_debug["scene"], "mode_red")
        self.assertAlmostEqual(baseline_debug["x"], 1.0, places=6)
        self.assertEqual(
            baseline_debug["reason"], "FIRST_TRANSITION_SET_AS_BASELINE"
        )
        self.assertFalse(baseline_debug["topic_sent"])

        publish_debug = json.loads(debug_pub.messages[1].data)
        self.assertEqual(publish_debug["mapped_type"], "spear_done_continue")
        self.assertEqual(publish_debug["device_id"], 3)
        self.assertEqual(publish_debug["raw_byte"], 0x11)
        self.assertEqual(publish_debug["mapped_byte"], 0x01)
        self.assertEqual(publish_debug["scene"], "mode_red")
        self.assertAlmostEqual(publish_debug["x"], 1.0, places=6)
        self.assertEqual(publish_debug["reason"], "MAPPED")
        self.assertTrue(publish_debug["topic_sent"])

    def test_snapshot_status_includes_shared_raw_byte(self) -> None:
        node = DummyNode()
        layer = self._build_layer(node)

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
        layer._handle_frame_locked(
            InfraredFrame(
                rx_stamp=Time(nanoseconds=1_020_000_000),
                device_id=3,
                report_type=0x01,
                raw_byte=0x22,
                device_timestamp_ms=120,
            )
        )
        layer._handle_frame_locked(
            InfraredFrame(
                rx_stamp=Time(nanoseconds=1_030_000_000),
                device_id=3,
                report_type=0x01,
                raw_byte=0x11,
                device_timestamp_ms=130,
            )
        )

        snapshot = layer.snapshot_status()
        self.assertEqual(snapshot["shared_last_event"]["raw_byte"], 0x11)
        self.assertEqual(snapshot["shared_last_event"]["source_device_id"], 3)
        self.assertEqual(snapshot["shared_last_event"]["aligned_ts_ms"], 1030)

    def test_claim_next_query_device_id_respects_poll_interval(self) -> None:
        node = DummyNode()
        layer = self._build_layer(node)
        first = layer.claim_next_query_device_id()
        second = layer.claim_next_query_device_id()
        self.assertEqual(first, 3)
        self.assertIsNone(second)

    def test_query_frame_crc_matches_modbus_sample(self) -> None:
        node = DummyNode()
        layer = self._build_layer(node)

        self.assertEqual(
            layer.build_query_frame(3),
            bytes([0x5A, 0xA5, 0x02, 0x03, 0x43, 0xBE]),
        )


if __name__ == "__main__":
    unittest.main()
