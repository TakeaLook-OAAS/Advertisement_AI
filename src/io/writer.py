import cv2

class VideoWriter:
    def __init__(self, path, fps, frame_size):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.w = cv2.VideoWriter(path, fourcc, float(fps), frame_size)

    def write(self, frame):
        self.w.write(frame)

    def release(self):
        self.w.release()
