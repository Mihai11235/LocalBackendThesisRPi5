import numpy as np
from pycoral.adapters import common, segment
from pycoral.utils.edgetpu import make_interpreter

class LaneSegmenter:
    def __init__(self, model_path="model/model_quant_edgetpu.tflite"):
        self.interpreter = make_interpreter(model_path, device=':0')
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.width, self.height = common.input_size(self.interpreter)

    def predict(self, input_data):
        scale, zero_point = self.input_details['quantization']
        input_quant = (input_data / scale + zero_point).astype(np.uint8)
        common.set_input(self.interpreter, input_quant)
        self.interpreter.invoke()

        output = segment.get_output(self.interpreter).astype(np.float32)
        out_scale, out_zero = self.output_details['quantization']
        return out_scale * (output - out_zero)
    