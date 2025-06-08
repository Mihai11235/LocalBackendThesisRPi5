from flask import Flask
from flask_cors import CORS
from camera.camera_interface import Camera
from model.lane_segmenter import LaneSegmenter
from websocket_handler import setup_websocket

app = Flask(__name__)
CORS(app)

# Init camera + model
camera = Camera()
segmenter = LaneSegmenter()

# Attach WebSocket
setup_websocket(app, camera, segmenter)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
