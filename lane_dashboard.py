from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import time
from PIL import Image
from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter
from picamera2 import Picamera2
from picamera2.utils import Transform

app = Flask(__name__)

# Load EdgeTPU model
interpreter = make_interpreter("model/model_quant_edgetpu.tflite", device=':0')
interpreter.allocate_tensors()
width, height = common.input_size(interpreter)

# Set up camera
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(
    main={"size": (1280, 720), "format": "RGB888"},
    transform=Transform(),
    controls={"AfMode": 2}
))
picam2.start()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess(input, height, width):
    original_img = Image.fromarray(input).convert('RGB')
    original_array = np.array(original_img)
    resized_img = original_img.resize((width, height), Image.LANCZOS)
    input_data = np.asarray(resized_img).astype(np.float32) / 255.0
    return original_array, input_data

def postprocess(original_array, output):
    prob = sigmoid(output)
    mask = np.squeeze(prob)
    mask_resized = cv2.resize(mask, original_array.shape[:2][::-1])
    mask_bin = (mask_resized > 0.5).astype(np.uint8)
    return mask_bin

def add_overlay(original_array, mask_bin):
    green = np.zeros_like(original_array)
    green[:, :, 1] = 255
    overlay = (np.expand_dims(mask_bin, axis=2) * green).astype(np.uint8)
    blended = cv2.addWeighted(original_array, 1.0, overlay, 0.6, 0)
    return blended

# def generate_frames():
#     while True:
#         frame = picam2.capture_array()
#         original_array, input_data = preprocess(frame, height, width)

#         input_details = interpreter.get_input_details()[0]
#         scale, zero_point = input_details['quantization']
#         input_quant = (input_data / scale + zero_point).astype(np.uint8)
#         common.set_input(interpreter, input_quant)

#         output_details = interpreter.get_output_details()[0]
#         out_scale, out_zero = output_details['quantization']

#         interpreter.invoke()
#         output = segment.get_output(interpreter).astype(np.float32)
#         output = out_scale * (output - out_zero)

#         mask_bin = postprocess(original_array, output)
#         result = add_overlay(original_array, mask_bin)

#         _, buffer = cv2.imencode('.jpg', result)
#         frame_bytes = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def generate_frames():
    frame_width = 1280
    center_tolerance = 100  # pixels tolerance before triggering warning

    while True:
        frame = picam2.capture_array()
        original_array, input_data = preprocess(frame, height, width)

        # Inference
        input_details = interpreter.get_input_details()[0]
        scale, zero_point = input_details['quantization']
        input_quant = (input_data / scale + zero_point).astype(np.uint8)
        common.set_input(interpreter, input_quant)

        output_details = interpreter.get_output_details()[0]
        out_scale, out_zero = output_details['quantization']

        interpreter.invoke()
        output = segment.get_output(interpreter).astype(np.float32)
        output = out_scale * (output - out_zero)

        # Postprocess
        mask_bin = postprocess(original_array, output)
        overlay = add_overlay(original_array, mask_bin)

        # Calculate lane center
        nonzero = np.column_stack(np.where(mask_bin > 0))  # (y, x)
        if len(nonzero) > 0:
            avg_x = int(np.mean(nonzero[:, 1]))  # average x (horizontal) position
            image_center = frame_width // 2
            offset = abs(avg_x - image_center)

            if offset > center_tolerance:
                # Draw lane departure warning
                cv2.putText(
                    overlay, 'LANE DEPARTURE!',
                    (overlay.shape[1] // 4, overlay.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA
                )
        else:
            # No lane detected at all
            cv2.putText(
                overlay, 'NO LANE DETECTED',
                (overlay.shape[1] // 6, overlay.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA
            )

        # Encode for MJPEG streaming
        _, buffer = cv2.imencode('.jpg', overlay)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')



@app.route('/')
def index():
    return render_template_string('''
        <html>
            <head>
                <title>Lane Detection</title>
                <style>
                    body { margin: 0; background: black; text-align: center; }
                    img { width: 100vw; height: 100vh; object-fit: cover; }
                </style>
            </head>
            <body>
                <img src="/video_feed">
            </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
