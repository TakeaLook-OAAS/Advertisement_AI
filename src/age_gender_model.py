import cv2
import numpy as np
from openvino import Core

class AgeGenderModel:
    def __init__(self, model_path):
        core = Core()
        model = core.read_model(model=model_path)
        self.compiled_model = core.compile_model(model=model, device_name="CPU")

    def predict(self, face_img):
        resized = cv2.resize(face_img, (62, 62))
        input_image = resized.transpose((2, 0, 1))
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype(np.float32)

        results = self.compiled_model([input_image])

        age = int(np.squeeze(results[self.compiled_model.output("age_conv3")]) * 100)
        gender_prob = results[self.compiled_model.output("prob")][0]
        gender = "Female" if gender_prob[0] > 0.5 else "Male"

        return age, gender
