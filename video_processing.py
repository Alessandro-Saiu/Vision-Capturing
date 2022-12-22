import cv2
import kornia
import numpy as np
import torch

"""------------------------------------------------------------------------------------------------------------------"""

VIDEO_PATH = r"C:/Users/aless/Documents/Training/test_256.mp4"


class VideoReader(object):
    def __init__(self, video_path=""):
        self.video_path = video_path
        self.frames_tensor = []
        self.frames = 0

    def reset(self):
        self.frames_tensor = []

    def open_video(self):
        caption = cv2.VideoCapture(self.video_path)
        length = int(caption.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_counter = 0

        while length > 8:
            if frame_counter == 8:
                frame_counter = 0

            ret, frame = caption.read()

            if not ret:
                print("Reading done")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame, (224, 224))
            self.frames_tensor.append(frame_resized)
            frame_counter += 1

            if frame_counter == 8:
                yield self.frames_tensor
                self.reset()

            length -= 1

