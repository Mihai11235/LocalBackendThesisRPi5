import unittest
import numpy as np

from logic.lane_departure import LaneDepartureDetector

class TestLaneDepartureDetector(unittest.TestCase):

    def setUp(self):
        """Set up a new detector for each test."""
        self.detector = LaneDepartureDetector()
        # Use the exact resolution from the camera interface
        self.height, self.width = 720, 1280
        self.overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.image_center_x = self.width // 2  # This will be 640

    def test_no_departure_is_none(self):
        """The lane is centered, so detect() should return None."""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # A lane perfectly centered in the 1280px image
        mask[:, self.image_center_x - 300] = 1  # Left line at x=340
        mask[:, self.image_center_x + 300] = 1  # Right line at x=940

        result = self.detector.detect(mask, self.overlay)
        self.assertIsNone(result)

    def test_right_departure_is_detected(self):
        """The lane is shifted by 100 pixels, so detect() should return 'lane_departure'."""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # --- This mask creates a 100-pixel departure ---
        # The lane's center will be at x=540. The image center is 640.
        # The absolute difference is > 100, so the warning should trigger.
        left_line_x = 135
        right_line_x = 940
        mask[:, left_line_x] = 1
        mask[:, right_line_x] = 1

        result = self.detector.detect(mask, self.overlay)
        self.assertEqual(result, "lane_departure")

    def test_left_departure_is_detected(self):
        """The lane is shifted by 100 pixels, so detect() should return 'lane_departure'."""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # --- This mask creates a 100-pixel departure ---
        # The lane's center will be at x=540. The image center is 640.
        # The absolute difference is > 100, so the warning should trigger.
        left_line_x = 340
        right_line_x = 1145
        mask[:, left_line_x] = 1
        mask[:, right_line_x] = 1

        result = self.detector.detect(mask, self.overlay)
        self.assertEqual(result, "lane_departure")

    def test_no_lane_is_no_lane(self):
        """The mask is empty, so detect() should return 'no_lane'."""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        result = self.detector.detect(mask, self.overlay)
        self.assertEqual(result, "no_lane")