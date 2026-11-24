import unittest
from unittest.mock import patch
import numpy as np

import pose_detector

class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)
    def cpu(self):
        return self
    def numpy(self):
        return self.value

class FakeKeypoints:
    def __init__(self, xy, conf):
        self.xy = FakeTensor(xy)
        self.conf = FakeTensor(conf)

class FakeBoxes:
    id = FakeTensor([7])

class FakeResult:
    def __init__(self):
        self.keypoints = FakeKeypoints(
            [[(100,100),(90,90),(110,90),(80,100),(120,100),
              (80,130),(120,130),(70,160),(130,160),(60,190),(140,190),
              (90,220),(110,220),(90,270),(110,270),(90,320),(110,320)]],
            [[.9]*17])
        self.boxes = FakeBoxes()

class FakeModel:
    def __init__(self, *args, **kwargs):
        self.calls = 0
    def track(self, *args, **kwargs):
        self.calls += 1
        return [FakeResult()]

class PoseBoxVisualizationTests(unittest.TestCase):
    def test_pose_outputs_boxes_only_no_joint_or_skeleton_detections(self):
        with patch.dict('sys.modules', {'ultralytics': type('U', (), {'YOLO': FakeModel})}):
            detector = pose_detector.BodyPoseDetector('fake.pt', 'cpu')
        frame = np.zeros((400, 300, 3), dtype=np.uint8)
        detections = detector.process(frame, run_inference=True)
        names = {d['class_name'] for d in detections}
        self.assertEqual(names, {'Left arm', 'Right arm', 'Left leg', 'Right leg', 'Torso'})
        self.assertTrue(all(len(d['box']) == 4 for d in detections))
        self.assertFalse(any('wrist' in n.lower() or 'eye' in n.lower() for n in names))

    def test_cached_boxes_return_on_skipped_inference_frames(self):
        with patch.dict('sys.modules', {'ultralytics': type('U', (), {'YOLO': FakeModel})}):
            detector = pose_detector.BodyPoseDetector('fake.pt', 'cpu')
        frame = np.zeros((400, 300, 3), dtype=np.uint8)
        first = detector.process(frame, run_inference=True)
        cached = detector.process(frame, run_inference=False)
        self.assertEqual(first, cached)
        self.assertEqual(detector.model.calls, 1)

if __name__ == '__main__':
    unittest.main()
