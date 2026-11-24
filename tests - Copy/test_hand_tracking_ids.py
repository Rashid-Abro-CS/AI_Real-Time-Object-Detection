import unittest
from unittest.mock import patch
import hand_detector

class HandIdTests(unittest.TestCase):
    def test_recent_hand_keeps_id_with_small_movement_and_label_change(self):
        with patch.object(hand_detector, 'time') as fake_time:
            fake_time.time.return_value = 100.0
            detector = hand_detector.HandLandmarkDetector(enabled=False)
            detector._frame_marker = 1
            first = detector._assign_id('Left hand', (100, 100))
            fake_time.time.return_value = 100.2
            detector._frame_marker = 2
            second = detector._assign_id('Right hand', (130, 110))
            self.assertEqual(first, second)

    def test_handedness_corrected_for_mirrored_preview(self):
        self.assertEqual(hand_detector.HandLandmarkDetector.correct_handedness_label('Left'), 'Right')
        self.assertEqual(hand_detector.HandLandmarkDetector.correct_handedness_label('Right'), 'Left')

    def test_two_hands_in_same_frame_do_not_share_id(self):
        detector = hand_detector.HandLandmarkDetector(enabled=False)
        detector._frame_marker = 1
        left = detector._assign_id('Left hand', (100, 100))
        right = detector._assign_id('Right hand', (400, 100))
        self.assertNotEqual(left, right)

if __name__ == '__main__':
    unittest.main()
