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

# 이거 필요 없을듯
# 왜냐면 6DRepNet에 얼굴 찾기 로직이 이미 있음 그래서 역효과가 날 수도 있대
# 그래도 일단 남겨 놓음 나중에 얼굴 자르는거 필요할 수도 있고