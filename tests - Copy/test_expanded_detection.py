import datetime
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from tracker import class_rules, ObjectSessionManager
from scene_classifier import SceneSessionManager
from detector import canonical_class, merge_supplemental
from detection_filters import suppress_duplicate_people
from relationship import RelationshipManager


class FakeDB:
    def __init__(self):
        self.docs = []
    def save_detection_session(self, doc):
        self.docs.append(doc)
    def save_scene_session(self, doc):
        self.docs.append(doc)
    def save_relationship_session(self, doc):
        self.docs.append(doc)


def det(track_id, cls, conf, box):
    return {"track_id": track_id, "class_name": cls,
            "confidence": conf, "box": tuple(map(float, box))}


class ExpandedDetectionTests(unittest.TestCase):
    def test_phone_has_lower_class_threshold(self):
        self.assertLess(class_rules("Mobile phone")["min_confidence"],
                        class_rules("Cup")["min_confidence"])

    def test_body_part_has_lower_threshold_and_stability_rule(self):
        rules = class_rules("Human hand")
        self.assertLess(rules["min_confidence"], class_rules("Cup")["min_confidence"])
        self.assertEqual(rules["min_hits"], 3)
        self.assertLess(rules["min_hit_ratio"], 0.4)
        self.assertLess(class_rules("Left arm")["min_confidence"], class_rules("Cup")["min_confidence"])
        self.assertLess(class_rules("Left index finger")["min_confidence"], class_rules("Cup")["min_confidence"])

    def test_body_landmark_can_confirm(self):
        db = FakeDB()
        manager = ObjectSessionManager(db, run_id="body")
        start = datetime.datetime(2026, 11, 20, 12, 0, 0)
        frames = 20
        for i in range(frames):
            manager.update([det(2000010, "Left wrist", .4, (20, 20, 40, 40))],
                          start + datetime.timedelta(seconds=i * .1))
        manager.close_all()
        self.assertEqual(len(db.docs), 1)
        self.assertEqual(db.docs[0]["object_name"], "Left wrist")

    def test_phone_synonyms_canonicalize(self):
        self.assertEqual(canonical_class("Cell phone"), "mobile phone")
        self.assertEqual(canonical_class("Smartphone"), "mobile phone")

    def test_supplemental_duplicate_removed(self):
        primary = [det(1, "Mobile phone", .65, (10, 10, 100, 100))]
        supplemental = [det(1000001, "Cell phone", .72, (12, 12, 98, 98)),
                       det(1000002, "Remote", .6, (200, 200, 300, 300))]
        merged = merge_supplemental(primary, supplemental)
        self.assertEqual(len(merged), 2)
        self.assertIn("Remote", [d["class_name"] for d in merged])

    def test_duplicate_people_keeps_highest_confidence_box(self):
        low = det(1, "Man", .40, (10, 10, 100, 100))
        high = det(2, "Person", .85, (12, 12, 98, 98))
        kept, dropped = suppress_duplicate_people([low, high])
        self.assertEqual(dropped, 1)
        self.assertEqual(kept, [high])

    def test_object_session_closes_by_elapsed_time(self):
        db = FakeDB()
        manager = ObjectSessionManager(db, run_id="time")
        start = datetime.datetime(2026, 11, 20, 12, 0, 0)
        for i in range(15):
            manager.update([det(5, "Laptop", .8, (0, 0, 100, 80))],
                          start + datetime.timedelta(seconds=i * .1))
        manager.update([], start + datetime.timedelta(seconds=3.5))
        manager.update([], start + datetime.timedelta(seconds=3.6))
        manager.update([], start + datetime.timedelta(seconds=3.7))
        self.assertEqual(len(db.docs), 1)
        self.assertEqual(db.docs[0]["object_name"], "Laptop")

    def test_relationship_closes_by_elapsed_time(self):
        db = FakeDB()
        manager = RelationshipManager(db)
        start = datetime.datetime(2026, 11, 20, 12, 0, 0)
        objects = [{"class_name": "Person"}, {"class_name": "Mobile phone"}]
        manager.update(objects, start)
        manager.update(objects, start + datetime.timedelta(seconds=2.2))
        manager.update([], start + datetime.timedelta(seconds=5))
        manager.update([], start + datetime.timedelta(seconds=5.1))
        manager.update([], start + datetime.timedelta(seconds=5.2))
        self.assertEqual(len(db.docs), 1)
        self.assertEqual(set(db.docs[0]["objects"]), {"Person", "Mobile phone"})

    def test_scene_gap_closes_old_session(self):
        db = FakeDB()
        manager = SceneSessionManager(db)
        start = datetime.datetime(2026, 11, 20, 12, 0, 0)
        manager.update("Office", .8, start)
        manager.update("Office", .8, start + datetime.timedelta(seconds=2.5))
        manager.update(None, 0, start + datetime.timedelta(seconds=10))
        manager.update("Kitchen", .8, start + datetime.timedelta(seconds=11))
        manager.close_all()
        # Office had enough duration to save; Kitchen was too short.
        self.assertEqual(len(db.docs), 1)
        self.assertEqual(db.docs[0]["scene_name"], "Office")


if __name__ == "__main__":
    unittest.main()
