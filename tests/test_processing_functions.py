import unittest
import numpy as np
import base64

# Import the functions to be tested
from model.preprocess import preprocess
from model.postprocess import postprocess
from logic.overlay import add_overlay
from utils.encoding import encode_frame_to_base64

class TestProcessingFunctions(unittest.TestCase):

    def test_preprocess(self):
        """
        Tests if preprocessing correctly resizes and normalizes an image.
        """
        # Create a dummy 3-channel black image of size 100x100
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        target_height, target_width = 50, 50

        processed_data = preprocess(dummy_image, target_height, target_width)

        # Check if the output has the correct shape
        self.assertEqual(processed_data.shape, (target_height, target_width, 3))
        # Check if the data type is float32
        self.assertEqual(processed_data.dtype, np.float32)
        # Check if values are normalized (between 0 and 1)
        self.assertTrue(np.all(processed_data >= 0) and np.all(processed_data <= 1))

    def test_postprocess(self):
        """
        Tests if postprocessing correctly creates a binary mask from raw model output.
        """
        original_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Create a dummy model output where sigmoid would result in values > 0.5
        model_output = np.ones((50, 50), dtype=np.float32) * 5

        binary_mask = postprocess(original_frame, model_output)

        # Check if the mask is resized to the original frame's 2D shape
        self.assertEqual(binary_mask.shape, (100, 100))
        # Check if the mask is binary (contains only 0s and 1s)
        self.assertTrue(np.all(np.unique(binary_mask) == [0, 1]) or np.all(binary_mask == 1))
        # Check if the data type is uint8
        self.assertEqual(binary_mask.dtype, np.uint8)

    def test_add_overlay(self):
        """
        Tests that the overlay function modifies the image where the mask is active.
        """
        original = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 5] = 1 # Set one pixel in the mask to 1

        overlayed_image = add_overlay(original.copy(), mask)

        # The pixel at (5, 5) should have changed color and not be [0, 0, 0]
        self.assertFalse(np.array_equal(overlayed_image[5, 5], [0, 0, 0]))
        # A pixel outside the mask should remain unchanged
        self.assertTrue(np.array_equal(overlayed_image[0, 0], [0, 0, 0]))
        # Check output type
        self.assertEqual(overlayed_image.dtype, np.uint8)

    def test_encode_frame_to_base64(self):
        """
        Tests if a frame is correctly encoded to a Base64 string.
        """
        dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        encoded_string = encode_frame_to_base64(dummy_frame)

        # Check if the output is a string
        self.assertIsInstance(encoded_string, str)
        # Check if the string is valid Base64 by trying to decode it
        try:
            base64.b64decode(encoded_string)
        except Exception as e:
            self.fail(f"Base64 decoding failed with error: {e}")

if __name__ == '__main__':
    unittest.main()