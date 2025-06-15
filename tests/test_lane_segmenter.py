import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from model.lane_segmenter import LaneSegmenter

class TestLaneSegmenter(unittest.TestCase):

    # @patch intercepts calls to the pycoral library and replaces them with mock objects.
    # These mocks are then passed as arguments to our test method.
    @patch('model.lane_segmenter.segment.get_output')
    @patch('model.lane_segmenter.common.set_input')
    @patch('model.lane_segmenter.common.input_size', return_value=(256, 256))
    @patch('model.lane_segmenter.make_interpreter')
    def test_predict_method_logic(self, mock_make_interpreter, mock_input_size, mock_set_input, mock_get_output):
        """
        Tests the quantization, invocation, and dequantization logic of the predict method.
        """
        # --- 1. ARRANGE (Configure the Mocks) ---

        # Create a mock interpreter object that our fake make_interpreter will return
        mock_interpreter = MagicMock()

        # Configure the mock get_input_details to return fake quantization parameters
        mock_interpreter.get_input_details.return_value = [{
            'quantization': (1.0, 0)  # scale=1.0, zero_point=0 for simple math
        }]

        # Configure the mock get_output_details similarly
        mock_interpreter.get_output_details.return_value = [{
            'quantization': (2.0, 5)  # scale=2.0, zero_point=5
        }]

        # Set our main mock function to return the mock interpreter
        mock_make_interpreter.return_value = mock_interpreter

        # Configure the mock get_output to return a fake raw output from the model
        fake_raw_output = np.array([[10, 20]], dtype=np.uint8)
        mock_get_output.return_value = fake_raw_output

        # --- 2. ACT ---

        # Now, instantiate our LaneSegmenter. Its __init__ method will use our mocked functions.
        segmenter = LaneSegmenter(model_path="fake/path/model.tflite")

        # Create some dummy input data to pass to the predict method
        input_data = np.random.rand(256, 256, 3).astype(np.float32)

        # Call the predict method we want to test
        result = segmenter.predict(input_data)

        # --- 3. ASSERT ---

        # Assert that the core inference function was called exactly once
        mock_interpreter.invoke.assert_called_once()

        # Assert that the set_input function was called. We can even inspect
        # what it was called with to verify the quantization logic.
        mock_set_input.assert_called_once()

        # Assert that the dequantization logic is correct.
        # We manually perform the calculation that should happen inside predict().
        out_scale, out_zero = 2.0, 5
        expected_result = out_scale * (fake_raw_output.astype(np.float32) - out_zero)

        # Check if the result from the predict method matches our expected result
        np.testing.assert_array_equal(result, expected_result)


if __name__ == '__main__':
    unittest.main()