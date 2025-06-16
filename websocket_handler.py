import json
import time
import threading
from flask_sock import Sock
from model.preprocess import preprocess
from model.postprocess import postprocess
from logic.overlay import add_overlay
from utils.encoding import encode_frame_to_base64
from logic.lane_departure import LaneDepartureDetector

# Initialize the WebSocket object
sock = Sock()

# Global variables to manage streaming state and thread
is_streaming = False
stream_thread = None

# Mutex for thread-safe operations
mutex = threading.Lock()

def setup_websocket(app, camera, segmenter):
    """
    Sets up WebSocket communication for the Flask application.

    This function initializes the WebSocket, configures routes for starting and stopping the stream,
    and handles WebSocket connections for streaming video frames and lane departure warnings.

    Args:
        app (Flask): The Flask application instance.
        camera (Camera): The camera interface for capturing video frames.
        segmenter (LaneSegmenter): The lane segmentation model for processing video frames.
    """
    sock.init_app(app)
    detector = LaneDepartureDetector()

    @app.route("/stop", methods=["POST"])
    async def stop_stream():
        """
        Stops the video streaming manually via the `/stop` endpoint.

        This endpoint sets the `is_streaming` flag to `False` to stop the stream.

        Returns:
            tuple: A response message and HTTP status code.
        """
        print("🔴 Trying to acquire mutex")
        with mutex:
            global is_streaming
            is_streaming = False

        print("🔴 Streaming manually stopped via /stop")
        return "Stopped", 200

    @sock.route('/ws')
    def websocket_endpoint(ws):
        """
        Handles WebSocket connections for streaming video frames and warnings.

        This function manages the streaming loop, processes video frames, and sends
        lane departure warnings to the connected WebSocket client.

        Args:
            ws (WebSocket): The WebSocket connection object.
        """
        global is_streaming, stream_thread
        print("Client connected")

        def stream_loop():
            """
            The main loop for streaming video frames and warnings.

            Captures frames from the camera, preprocesses them, performs lane segmentation,
            and sends the processed frames and warnings to the WebSocket client.
            """
            global is_streaming
            print("🟢 Streaming started")

            while True:
                with mutex:
                    if not is_streaming:
                        print("🔴 Streaming stopped")
                        break

                frame = camera.capture_frame()
                input_data = preprocess(frame, segmenter.height, segmenter.width)
                output = segmenter.predict(input_data)
                mask_bin = postprocess(frame, output)
                overlay = add_overlay(frame, mask_bin)
                warning = detector.detect(mask_bin, overlay)

                encoded = encode_frame_to_base64(overlay)

                try:
                    ws.send(json.dumps({"type": "frame", "image": encoded}))
                    if warning == "lane_departure":
                        ws.send(json.dumps({"type": "warning", "status": warning}))
                    time.sleep(1 / 34)
                except:
                    print("⚠️ Client disconnected during stream")
                    break

        try:
            while True:
                msg = ws.receive()
                if msg == "start":
                    with mutex:
                        if not is_streaming:
                            is_streaming = True
                            stream_thread = threading.Thread(target=stream_loop)
                            stream_thread.start()
                elif msg in ["stop", "exit"]:
                    with mutex:
                        is_streaming = False
                        if msg == "exit":
                            break
        except Exception as e:
            print("WebSocket error:", e)
            with mutex:
                is_streaming = False
