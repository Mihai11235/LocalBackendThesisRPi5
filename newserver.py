import cv2
import numpy as np
import time
import base64
import json
from pycoral.adapters import common, segment
from pycoral.utils.edgetpu import make_interpreter
from picamera2 import Picamera2
from picamera2.utils import Transform
from flask import Flask, render_template, Response
from flask_cors import CORS

from flask_sock import Sock
import threading

from fake_camera import FakeCamera

app = Flask(__name__)
sock = Sock(app)
CORS(app)

# Load EdgeTPU model1
interpreter = make_interpreter("model1/model_quant_edgetpu.tflite", device=':0')
interpreter.allocate_tensors()
width, height = common.input_size(interpreter)

# Set up camera
picam2 = Picamera2()
# picam2 = FakeCamera("test_video.mp4")
picam2.configure(picam2.create_video_configuration(
    main={"size": (1280, 720), "format": "RGB888"},
    transform=Transform(),
    controls={"AfMode": 2}
))
picam2.start()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def preprocess(input, height, width):
    # Resize with OpenCV directly
    resized = cv2.resize(input, (width, height), interpolation=cv2.INTER_LINEAR)

    # Convert to float32 and normalize
    input_data = resized.astype(np.float32) / 255.0

    return input, input_data  # input is already the original_array


def postprocess(original_array, output):
    prob = sigmoid(output)
    mask = np.squeeze(prob)
    # mask_resized = cv2.resize(mask, original_array.shape[:2][::-1])
    mask_resized = cv2.resize(prob, original_array.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    mask_bin = (mask_resized > 0.5).astype(np.uint8)
    return mask_bin

def add_overlay(original, mask_bin):
    # Reuse input image and modify a copy directly
    overlay = original.copy()

    # Apply green tint only where mask is 1 (vectorized boolean indexing)
    overlay[mask_bin == 1] = overlay[mask_bin == 1] * 0.4 + np.array([0, 153, 0])  # more efficient than full green

    return overlay.astype(np.uint8)


@app.route("/stop", methods=["POST"])
async def stop_stream():
    print("🔴 Trying to acquire mutex")
    with mutex:
        global is_streaming
        is_streaming = False

    print("🔴 Streaming manually stopped via /stop")
    return "Stopped", 200


def detect_lane_departure(mask_bin, overlay):
    img = mask_bin

    row_sums = img.sum(axis=1)

    # Find top and bottom rows where lane exists
    nonzero_rows = np.where(row_sums > 0)[0]
    if len(nonzero_rows) == 0:
        print("No lane detected")
        cv2.putText(overlay, "No lane detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 1), 2)
        return "no_lane"
    top = nonzero_rows[0]
    bottom = nonzero_rows[-1]

    height, width = img.shape
    cv2.line(overlay, (0, top), (width, top), (1, 1), 2)  # White vertical line
    cv2.line(overlay, (0, bottom), (width, bottom), (1, 1), 2)  # White vertical line

    middle_x = width // 2
    coord_y = (3 * top + 5 * bottom) // 8

    # Measure distance from the line to the first green pixel on both sides (left and right)
    def find_first_white_pixel(image, middle_x, height, direction='left'):
        """Finds the first green pixel on the specified side of the vertical line, based on height."""
        if direction == 'left':
            for x in range(middle_x, -1, -1):  # Going left from the middle_x
                # Check if green channel is dominant and red/blue are minimal
                if image[height, x] == 1:
                    return middle_x - x, x  # Distance from the vertical line
        elif direction == 'right':
            for x in range(middle_x, width):  # Going right from the middle_x
                # Check if green channel is dominant and red/blue are minimal
                if image[height, x] == 1:
                    return x - middle_x, x  # Distance from the vertical line

        return None, None  # No white pixel found

    # Get distances from the line
    distance_left, coord_left = find_first_white_pixel(img, middle_x, coord_y, direction='left')
    distance_right, coord_right = find_first_white_pixel(img, middle_x, coord_y, direction='right')

    if distance_left is None and distance_right is None:
        print("Cannot detect lane lines")
        cv2.putText(overlay, "Cannot detect lane lines", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 1), 2)
        return "no_lane"
    elif distance_left is None or distance_right is None:
        print("No lane detected!")
        cv2.putText(overlay, "No Lane Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 1), 2)
        return "no_lane"

    lane_middle = (coord_left + coord_right) // 2
    cv2.line(overlay, (lane_middle, 0), (lane_middle, height), (1, 1), 2)  # White vertical line

    cv2.line(overlay, (middle_x, 0), (middle_x, height), (1, 1), 2)  # White vertical line
    cv2.line(overlay, (0, coord_y), (width, coord_y), (1, 1), 2)

    # print(abs(middle_x - lane_middle) // 2)

    if abs(middle_x - lane_middle) // 2 > 50:
        print("Lane departure warning!")
        cv2.putText(overlay, "Lane Departure Warning!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 1), 2)
        return "lane_departure"

    return None


is_streaming = False
stream_thread = None
mutex = threading.Lock()


def stream_loop(ws):
    global is_streaming
    print("🟢 Streaming started")

    while True:
        with mutex:
            if not is_streaming:
                print("🔴 Streaming stopped")
                break

        frame = picam2.capture_array()
        prev_time = time.time()
        original_array, input_data = preprocess(frame, height, width)

        input_details = interpreter.get_input_details()[0]
        scale, zero_point = input_details['quantization']

        # Quantize input
        input_quant = (input_data / scale + zero_point).astype(np.uint8)
        common.set_input(interpreter, input_quant)

        output_details = interpreter.get_output_details()[0]
        out_scale, out_zero = output_details['quantization']

        interpreter.invoke()

        # Dequantize output
        output = segment.get_output(interpreter).astype(np.float32)
        output = out_scale * (output - out_zero)

        mask_bin = postprocess(original_array, output)
        overlay = add_overlay(original_array, mask_bin)

        warning = detect_lane_departure(mask_bin, overlay)

        _, buffer = cv2.imencode('.jpg', overlay)
        frame_bytes = base64.b64encode(buffer).decode('utf-8')


        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        # print(f"Time taken: {curr_time - prev_time:.5f} seconds")
        prev_time = curr_time

        print(f"FPS: {fps:.2f}")
        try:
            time.sleep(1 / 34)
            ws.send(json.dumps({"type": "frame", "image": frame_bytes}))

            if warning == "lane_departure":
                ws.send(json.dumps({"type": "warning", "status": warning}))
        except:
            print("⚠️ Client disconnected during stream")
            break


@sock.route('/ws')
def websocket_endpoint(ws):
    global is_streaming, stream_thread

    print("Client connected")

    try:
        while True:
            message = ws.receive()
            print(f"📩 Message: {message}")

            if message == "start":
                with mutex:
                    if not is_streaming:
                        is_streaming = True
                        stream_thread = threading.Thread(target=stream_loop, args=(ws,))
                        stream_thread.start()
            elif message == "stop":
                with mutex:
                    is_streaming = False

            elif message == "exit":
                with mutex:
                    is_streaming = False
                print("❌ Client exited")
                break

    except Exception as e:
        print("WebSocket error:", e)
        with mutex:
            is_streaming = False


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)