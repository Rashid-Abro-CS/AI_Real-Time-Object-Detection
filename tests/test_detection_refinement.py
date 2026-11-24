import unittest
import time

from detector import ObjectDetector, canonical_class
from config import CLASS_RULES
from tracker import ObjectSessionManager


class RefinementTests(unittest.TestCase):
    def test_tiles_overlap_and_cover_large_frame(self):
        class Frame:
            shape = (700, 1250, 3)
        tiles = ObjectDetector._tiles(Frame())
        self.assertEqual(len(tiles), 4)
        self.assertEqual(tiles[0][0], 0)
        self.assertEqual(tiles[-1][2], 1250)
        self.assertEqual(tiles[-1][3], 700)

    def test_head_crop_is_upper_person_region(self):
        frame = type("Frame", (), {"shape": (700, 1250, 3)})()
        person = {"class_name": "Person", "confidence": .8, "box": (300, 100, 500, 600)}
        detector = ObjectDetector.__new__(ObjectDetector)
        crops = detector._head_crops(frame, [person])
        self.assertEqual(len(crops), 1)
        x1, y1, x2, y2 = crops[0]
        self.assertLess(y2, 300)
        self.assertGreater(x2-x1, 200)

    def test_refinement_tracker_keeps_ids_and_separates_two_same_class_objects(self):
        det = ObjectDetector.__new__(ObjectDetector)
        det._refinement_tracks = {}
        det._next_refinement_id = 1_500_000
        primary = []
        candidates = [
            {"class_name": "Glasses", "confidence": .4, "box": (100, 100, 180, 130)},
            {"class_name": "Glasses", "confidence": .4, "box": (300, 100, 380, 130)},
        ]
        first = det._track_and_dedupe_refinements(candidates, primary, time.monotonic(), 4)
        self.assertEqual(len(first), 2)
        self.assertNotEqual(first[0]["track_id"], first[1]["track_id"])
        second_candidates = [dict(c, box=(c["box"][0]+3, c["box"][1], c["box"][2]+3, c["box"][3])) for c in candidates]
        second = det._track_and_dedupe_refinements(second_candidates, primary, time.monotonic(), 8)
        self.assertEqual({d["track_id"] for d in first}, {d["track_id"] for d in second})

    def test_crop_on_face_hint_can_confirm_lower_confidence_eyewear(self):
        manager = ObjectSessionManager(db=None, run_id="test")
        t0 = time.time()
        confirmed = []
        for i in range(8):
            confirmed = manager.update([{
                "track_id": 9001, "class_name": "Glasses", "confidence": .22,
                "box": (200, 100, 280, 135), "context_hint": "crop_on_face",
            }], timestamp=__import__("datetime").datetime.fromtimestamp(t0+i*.1))
        self.assertTrue(any(d["class_name"] == "Glasses" for d in confirmed))

    def test_glasses_confidence_rule_is_lower_than_generic(self):
        self.assertLess(CLASS_RULES["glasses"]["min_confidence"], .35)
        self.assertEqual(canonical_class("Eyeglasses"), "glasses")


if __name__ == "__main__":
    unittest.main()
