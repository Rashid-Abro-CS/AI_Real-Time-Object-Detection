import unittest
from unittest.mock import patch
import numpy as np
import hand_detector

class HandROITests(unittest.TestCase):
    def test_full_frame_and_wrist_crops_are_run_and_duplicates_merged(self):
        detector = hand_detector.HandLandmarkDetector(enabled=False)
        detector.available = True
        detector._frame_marker = 0
        # 21 points with a bounding area; first full-frame and crop results overlap.
        pts1 = [(100+i%5, 100+i//5) for i in range(21)]
        pts2 = [(105+i%5, 105+i//5) for i in range(21)]
        calls = []
        def fake_infer(image, offset=(0, 0), scale=1.0):
            calls.append((image.shape, offset, scale))
            return [{"side":"Left hand", "confidence":0.8, "points":pts1}] if len(calls)==1 else [
                {"side":"Left hand", "confidence":0.9, "points":pts2}]
        detector._infer_on_image = fake_infer
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch.object(hand_detector.time, "time", return_value=100.0):
            detections = detector.process(frame, wrist_hints=[(100, 100), (300, 200)])
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(detector.last_count, 1)  # duplicate crop/full detection merged
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class_name"], "Left Hand")

    def test_no_wrist_hints_still_runs_full_frame_detector(self):
        detector = hand_detector.HandLandmarkDetector(enabled=False)
        detector.available = True
        called = []
        detector._infer_on_image = lambda image, **kw: called.append(1) or []
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        detector.process(frame, wrist_hints=[])
        self.assertEqual(called, [1])
        self.assertEqual(detector.last_count, 0)

if __name__ == "__main__":
    unittest.main()
