import numpy as np
from pycoral.adapters import common, segment
from pycoral.utils.edgetpu import make_interpreter

class LaneSegmenter:
    """
    A class for lane segmentation using a TensorFlow Lite model optimized for Edge TPU.

    Attributes:
        interpreter (pycoral.utils.edgetpu.Interpreter): The Edge TPU interpreter for the model.
        input_details (dict): Details about the model's input tensor.
        output_details (dict): Details about the model's output tensor.
        width (int): Width of the model's input image.
        height (int): Height of the model's input image.
    """
    def __init__(self, model_path="model/model_quant_edgetpu.tflite"):
        """
        Initializes the LaneSegmenter with the specified model.

        Args:
            model_path (str): Path to the TensorFlow Lite model file. Defaults to "model/model_quant_edgetpu.tflite".
        """
        self.interpreter = make_interpreter(model_path, device=':0')
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.width, self.height = common.input_size(self.interpreter)

    def predict(self, input_data):
        """
        Performs lane segmentation on the input data.

        The input data is quantized and passed to the Edge TPU interpreter for inference.
        The output is dequantized and returned as a NumPy array.

        Args:
            input_data (numpy.ndarray): The input image data as a NumPy array.

        Returns:
            numpy.ndarray: The dequantized output of the model, representing lane segmentation.
        """
        scale, zero_point = self.input_details['quantization']
        input_quant = (input_data / scale + zero_point).astype(np.uint8)
        common.set_input(self.interpreter, input_quant)
        self.interpreter.invoke()

        output = segment.get_output(self.interpreter).astype(np.float32)
        out_scale, out_zero = self.output_details['quantization']
        return out_scale * (output - out_zero)
    