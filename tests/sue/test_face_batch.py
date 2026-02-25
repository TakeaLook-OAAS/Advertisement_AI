#여러 이미지를 한번에 처리
import cv2
import numpy as np
from openvino import Core
import os
import glob

model_path = "models/face_detection/face-detection-adas-0001.xml"
core = Core()
model = core.read_model(model=model_path)
compiled_model = core.compile_model(model=model, device_name="CPU")
output_layer = compiled_model.output(0)

# 폴더 경로
image_folder = "data/samples/test_images"
crop_folder = "data/output/crop_faces"
result_folder = "data/output/result_images"

os.makedirs(crop_folder, exist_ok=True)
os.makedirs(result_folder, exist_ok=True)

# 이미지 리스트
image_paths = []
for ext in ["jpg", "png", "jpeg"]:
    image_paths.extend(glob.glob(f"{image_folder}/*.{ext}"))

for img_idx, img_path in enumerate(image_paths):
    image = cv2.imread(img_path)
    if image is None:
        print(f"{img_path} 로드 실패, 스킵")
        continue

    orig_h, orig_w = image.shape[:2]

    # 모델 입력 크기로 변환
    resized = cv2.resize(image, (672, 384))
    input_image = resized.transpose((2, 0, 1))
    input_image = np.expand_dims(input_image, axis=0)

    # 얼굴 탐지 실행
    results = compiled_model([input_image])[output_layer]

    # Detection + Crop 저장
    for idx, detection in enumerate(results[0][0]):
        confidence = float(detection[2])
        if confidence > 0.5:
            xmin = int(detection[3] * orig_w)
            ymin = int(detection[4] * orig_h)
            xmax = int(detection[5] * orig_w)
            ymax = int(detection[6] * orig_h)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # 얼굴 Crop
            face_img = image[ymin:ymax, xmin:xmax]
            crop_path = f"{crop_folder}/crop_{img_idx}_{idx}.jpg" #예: crop_0_0.jpg  (0번 이미지의 0번 얼굴)
            cv2.imwrite(crop_path, face_img)

    # 전체 이미지 결과 저장
    result_path = f"{result_folder}/result_{img_idx}.jpg"
    cv2.imwrite(result_path, image)
    print(f"{img_path} 처리 완료 → {result_path}")
