import cv2
import numpy as np
from openvino import Core

class FaceDetector:
    def __init__(self, model_path):
        core = Core()
        model = core.read_model(model=model_path)
        self.compiled_model = core.compile_model(model=model, device_name="CPU")
        self.output_layer = self.compiled_model.output(0)

    def detect(self, image):
        orig_h, orig_w = image.shape[:2]

        resized = cv2.resize(image, (672, 384))
        input_image = resized.transpose((2, 0, 1))
        input_image = np.expand_dims(input_image, axis=0)

        results = self.compiled_model([input_image])[self.output_layer]

        boxes = []
        for detection in results[0][0]:
            confidence = float(detection[2])
            if confidence > 0.5:
                xmin = int(detection[3] * orig_w)
                ymin = int(detection[4] * orig_h)
                xmax = int(detection[5] * orig_w)
                ymax = int(detection[6] * orig_h)
                boxes.append((xmin, ymin, xmax, ymax))

        return boxes
