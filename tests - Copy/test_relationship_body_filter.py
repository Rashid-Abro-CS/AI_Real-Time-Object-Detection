import unittest
import datetime
from relationship import RelationshipManager

class FakeDB:
    def __init__(self):
        self.saved = []
    def save_relationship_session(self, doc):
        self.saved.append(doc)

class RelationshipBodyFilterTests(unittest.TestCase):
    def test_body_parts_do_not_form_scene_relationships(self):
        db = FakeDB()
        manager = RelationshipManager(db)
        manager.update([
            {'class_name': 'Person'},
            {'class_name': 'Left arm'},
            {'class_name': 'Torso'},
        ], timestamp=datetime.datetime(2026,1,1,12,0,0))
        self.assertEqual(manager.current_group(), [])
        self.assertEqual(manager._active, {})

if __name__ == '__main__':
    unittest.main()
