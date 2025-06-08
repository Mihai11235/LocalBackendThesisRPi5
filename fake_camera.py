import cv2

class FakeCamera:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

    def configure(self, *_args, **_kwargs):
        pass

    def start(self):
        pass

    def create_video_configuration(self, *_args, **_kwargs):
        return None

    def capture_array(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return frame

    def release(self):
        self.cap.release()
