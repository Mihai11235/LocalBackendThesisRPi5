import cv2

class FakeCamera:
    """
    A class to simulate a camera using a video file.

    This class provides methods to interact with a video file as if it were a live camera feed.
    It allows capturing frames, resetting the video playback, and releasing resources.

    Attributes:
        cap (cv2.VideoCapture): OpenCV video capture object for the video file.
    """
    def __init__(self, video_path):
        """
        Initializes the FakeCamera with the specified video file.

        Args:
            video_path (str): Path to the video file to be used as the camera feed.
        """
        self.cap = cv2.VideoCapture(video_path)

    def configure(self, *_args, **_kwargs):
        pass

    def start(self):
        pass

    def create_video_configuration(self, *_args, **_kwargs):
        return None

    def capture_array(self):
        """
        Captures a frame from the video file.

        If the end of the video is reached, the playback is reset to the beginning.

        Returns:
            numpy.ndarray: The captured video frame as a NumPy array.
        """
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return frame

    def release(self):
        """
        Releases the video capture object.

        This method frees the resources associated with the video file.
        """
        self.cap.release()
