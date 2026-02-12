import cv2
import numpy as np
from openvino import Core
import os

model_path = "models/face_detection/ir/face-detection-adas-0001.xml"
core = Core()
model = core.read_model(model=model_path)
compiled_model = core.compile_model(model=model, device_name="CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

image = cv2.imread("data/test.jpg")
if image is None:
    raise ValueError("data/test.jpg 파일이 없습니다.")

orig_h, orig_w = image.shape[:2]

# 모델 입력 크기 (W=672, H=384)
resized = cv2.resize(image, (672, 384))
input_image = resized.transpose((2, 0, 1))  # HWC → CHW
input_image = np.expand_dims(input_image, axis=0)

# 모델 실행, 얼굴 탐지 결과
results = compiled_model([input_image])[output_layer]

# 결과 처리 및 crop 저장
os.makedirs("data/crop_faces", exist_ok=True)

for idx, detection in enumerate(results[0][0]):
    confidence = float(detection[2])
    if confidence > 0.5:
        xmin = int(detection[3] * orig_w)
        ymin = int(detection[4] * orig_h)
        xmax = int(detection[5] * orig_w)
        ymax = int(detection[6] * orig_h)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # 얼굴 crop
        face_img = image[ymin:ymax, xmin:xmax]

        # 파일 저장
        crop_path = f"data/crop_faces/crop_face_{idx}.jpg" #예: crop_face_0.jpg
        cv2.imwrite(crop_path, face_img)
        print(f"crop 얼굴 저장: {crop_path}")

cv2.imwrite("data/result.jpg", image)
print("Detection 완료! 전체 이미지: data/result.jpg")