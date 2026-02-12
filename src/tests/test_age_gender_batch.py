import cv2
import numpy as np
from openvino import Core
import os

# 모델 경로
model_path = "models/age_gender/ir/age-gender-recognition-retail-0013.xml"

# OpenVINO Core 생성
core = Core()
model = core.read_model(model=model_path)
compiled_model = core.compile_model(model=model, device_name="CPU")

# Crop 이미지 폴더
crop_folder = "data/crop_faces/"
results_folder = "data/age_gender_results/"
os.makedirs(results_folder, exist_ok=True)

# batch 처리
for img_name in os.listdir(crop_folder):
    img_path = os.path.join(crop_folder, img_name)
    image = cv2.imread(img_path)
    if image is None:
        continue

    # 모델 입력 크기: (C,H,W) = (3,62,62)
    resized = cv2.resize(image, (62, 62))
    input_image = resized.transpose((2,0,1))
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype(np.float32)

    # 추론
    results = compiled_model([input_image])

    # 레이어별 결과 가져오기
    age = int(np.squeeze(results[compiled_model.output("age_conv3")]) * 100)
    gender_prob = results[compiled_model.output("prob")][0] #[[0.8, 0.2]]
    gender = "Female" if gender_prob[0] > 0.5 else "Male"

    print(f"{img_name}: Age ~ {age}, Gender: {gender}")

    # 이미지 저장 (원하면)
    cv2.putText(image, f"{gender}, {age}", (5,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imwrite(os.path.join(results_folder, img_name), image)

print("Batch Age/Gender 추론 완료!")
