
import cv2


class WebcamCamera:

    def __init__(self, camera_index=0, width=1280, height=720):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

    def connect(self):

        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open laptop webcam. "
                "Check camera permissions or try another camera index."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        return True

    def start(self):
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Webcam is not connected.")

    def get_frame(self):

        if self.cap is None or not self.cap.isOpened():
            return None

        success, frame = self.cap.read()

        if not success:
            return None

        return frame

    def stop(self):

        if self.cap is not None:
            self.cap.release()
            self.cap = None
