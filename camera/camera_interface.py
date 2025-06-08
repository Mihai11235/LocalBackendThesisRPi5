from picamera2 import Picamera2
from picamera2.utils import Transform

class Camera:
    def __init__(self):
        self.cam = Picamera2()
        self.cam.configure(self.cam.create_video_configuration(
            main={"size": (1280, 720), "format": "RGB888"},
            transform=Transform(), controls={"AfMode": 2}
        ))
        self.cam.start()

    def capture_frame(self):
        return self.cam.capture_array()
