from __future__ import annotations

import json
import math
import unittest

from _module_stubs import install_test_stubs

install_test_stubs()

from rclpy.time import Time

from agv_pose_refiner_py.common import CoarsePose, SolveResult
from agv_pose_refiner_py.result_publisher import ResultPublishLayer


class _Header:
    def __init__(self) -> None:
        self.frame_id = ""
        self.stamp = None


class DummyPoseMsg:
    def __init__(self) -> None:
        self.header = _Header()
        self.source = ""
        self.string = ""
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0
        self.qw = 1.0


class DummyPublisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, msg: object) -> None:
        self.messages.append(msg)


class DummyNode:
    def __init__(self) -> None:
        self.publishers = {}

    def create_publisher(
        self, _msg_type: type, topic: str, _queue_size: int
    ) -> DummyPublisher:
        publisher = DummyPublisher()
        self.publishers[topic] = publisher
        return publisher


class TestResultPublisher(unittest.TestCase):
    def _build_layer(self, *, publish_tf: bool = True) -> ResultPublishLayer:
        return ResultPublishLayer(
            node=DummyNode(),
            pose_msg_type=DummyPoseMsg,
            refined_pose_topic="/pose/laser",
            status_topic="/laser_status",
            world_frame_id="map",
            tf_parent_frame="map",
            tf_child_frame="base_link",
            publish_tf=publish_tf,
        )

    def _make_coarse_pose(self) -> CoarsePose:
        return CoarsePose(
            stamp=Time(nanoseconds=0),
            x=1.0,
            y=2.0,
            z=0.5,
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_deg=0.0,
        )

    def _make_result(
        self,
        *,
        publish_x: float | None,
        publish_y: float | None,
        publish_z: float | None,
        publish_yaw_deg: float | None,
        yaw_deg: float = 15.0,
    ) -> SolveResult:
        return SolveResult(
            state="REFINED",
            reason="OK",
            pose_source="test",
            localized=True,
            x=1.2,
            y=2.3,
            yaw_deg=yaw_deg,
            valid_beam_count=2,
            score=1.0,
            prior_age_ms=0.0,
            publish_x=publish_x,
            publish_y=publish_y,
            publish_z=publish_z,
            publish_yaw_deg=publish_yaw_deg,
        )

    def test_publish_x_only_sets_unsolved_fields_to_nan(self) -> None:
        layer = self._build_layer()
        coarse = self._make_coarse_pose()
        result = self._make_result(
            publish_x=1.2,
            publish_y=None,
            publish_z=None,
            publish_yaw_deg=None,
        )

        layer.publish(coarse, result)

        pose_msg = layer._pose_pub.messages[-1]
        self.assertAlmostEqual(pose_msg.x, 1.2, places=6)
        self.assertTrue(math.isnan(pose_msg.y))
        self.assertTrue(math.isnan(pose_msg.z))
        self.assertTrue(math.isnan(pose_msg.qx))
        self.assertTrue(math.isnan(pose_msg.qy))
        self.assertTrue(math.isnan(pose_msg.qz))
        self.assertTrue(math.isnan(pose_msg.qw))
        self.assertEqual(len(layer._tf_broadcaster.transforms), 0)

        status = json.loads(layer._status_pub.messages[-1].data)
        self.assertEqual(status["laser_pose_output_fields"], ["x"])
        self.assertFalse(status["laser_pose_output_complete"])

    def test_publish_xy_without_solved_yaw_sets_quaternion_to_nan(self) -> None:
        layer = self._build_layer()
        coarse = self._make_coarse_pose()
        result = self._make_result(
            publish_x=1.2,
            publish_y=2.3,
            publish_z=0.5,
            publish_yaw_deg=None,
        )

        layer.publish(coarse, result)

        pose_msg = layer._pose_pub.messages[-1]
        self.assertAlmostEqual(pose_msg.x, 1.2, places=6)
        self.assertAlmostEqual(pose_msg.y, 2.3, places=6)
        self.assertAlmostEqual(pose_msg.z, 0.5, places=6)
        self.assertTrue(math.isnan(pose_msg.qx))
        self.assertTrue(math.isnan(pose_msg.qy))
        self.assertTrue(math.isnan(pose_msg.qz))
        self.assertTrue(math.isnan(pose_msg.qw))
        self.assertEqual(len(layer._tf_broadcaster.transforms), 0)

        status = json.loads(layer._status_pub.messages[-1].data)
        self.assertEqual(status["laser_pose_output_fields"], ["x", "y", "z"])
        self.assertFalse(status["laser_pose_output_complete"])

    def test_publish_full_pose_uses_publish_yaw_and_broadcasts_tf(self) -> None:
        layer = self._build_layer()
        coarse = self._make_coarse_pose()
        result = self._make_result(
            publish_x=1.2,
            publish_y=2.3,
            publish_z=0.5,
            publish_yaw_deg=90.0,
            yaw_deg=15.0,
        )

        layer.publish(coarse, result)

        pose_msg = layer._pose_pub.messages[-1]
        self.assertAlmostEqual(pose_msg.x, 1.2, places=6)
        self.assertAlmostEqual(pose_msg.y, 2.3, places=6)
        self.assertAlmostEqual(pose_msg.z, 0.5, places=6)
        self.assertAlmostEqual(pose_msg.qx, 0.0, places=6)
        self.assertAlmostEqual(pose_msg.qy, 0.0, places=6)
        self.assertAlmostEqual(pose_msg.qz, math.sqrt(0.5), places=6)
        self.assertAlmostEqual(pose_msg.qw, math.sqrt(0.5), places=6)

        self.assertEqual(len(layer._tf_broadcaster.transforms), 1)
        transform = layer._tf_broadcaster.transforms[0]
        self.assertAlmostEqual(transform.transform.translation.x, 1.2, places=6)
        self.assertAlmostEqual(transform.transform.translation.y, 2.3, places=6)
        self.assertAlmostEqual(transform.transform.translation.z, 0.5, places=6)

        status = json.loads(layer._status_pub.messages[-1].data)
        self.assertEqual(
            status["laser_pose_output_fields"],
            ["x", "y", "z", "qx", "qy", "qz", "qw"],
        )
        self.assertTrue(status["laser_pose_output_complete"])


if __name__ == "__main__":
    unittest.main()
