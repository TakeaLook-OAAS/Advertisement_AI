# 이미지 입력 -> yolo detection -> face detection -> headpose -> eye detection -> gaze estimation
# headpose vector + gaze vector를 그려서 이미지로 저장

from __future__ import annotations
import os
import cv2
import numpy as np
from loguru import logger
from src.utils.types import BBoxXYXY, Track, Gaze
from src.models.yolo_detector import YoloDetector
from src.models.face_openvino import FaceDetector
from src.models.headpose_6drepnet import HeadPoseEstimator
from src.models.eye_openvino import EyeDetector
from src.models.gaze_openvino import GazeDetector
from src.vision.draw import draw_tracks, draw_crop_bbox, draw_headpose, draw_gaze


INPUT_IMAGE = "data/samples/test_8.jpg"
OUTPUT_DIR = "data/output/"
OUTPUT_IMAGE = "test_8_gaze_pipeline.jpg"

LEFT_IMAGE = "data/samples/test_eye/test_4_left.jpg"
RIGHT_IMAGE = "data/samples/test_eye/test_4_right.jpg"


def pipeline():     # 파이프라인대로 EyeDetector에서 Eye crop해서 입력
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) 이미지 로드
    frame = cv2.imread(INPUT_IMAGE)
    if frame is None:
        print(f"이미지를 읽을 수 없습니다: {INPUT_IMAGE}")
        return

    h, w = frame.shape[:2]
    print(f"이미지 크기(h, w): {h, w}")

    # 2) yolo detection (person)
    yolo = YoloDetector({"model": "weights/yolo/yolov8n.pt", "conf": 0.5, "classes": [0]})
    dets = yolo.detect(frame)
    if not dets:
        print("사람을 찾지 못했습니다.")
        return
    print(f"검출된 사람 수: {len(dets)}")

    # det -> track 변환
    tracks = [Track(track_id=i, bbox=d.bbox) for i, d in enumerate(dets)]

    # 3) face detection
    face_detector = FaceDetector({"device": "CPU"})
    tracks = face_detector.detect_batch(frame, tracks)

    # 4) headpose → track.headpose
    hp_estimator = HeadPoseEstimator({"device": "cpu"})
    tracks = hp_estimator.infer_batch(frame, tracks)
    for t in tracks:
        hp = t.headpose
        print(f"track {t.track_id} headpose: yaw={hp.yaw:+.1f}, pitch={hp.pitch:+.1f}, roll={hp.roll:+.1f}")

    # 5) eye detection
    eye_detector = EyeDetector({"device": "CPU"})
    tracks = eye_detector.detect_batch(frame, tracks)
    for t in tracks:
        print(f"track {t.track_id} left_eye: {t.left_eye}, right_eye: {t.right_eye}")

    # 6) gaze estimation → track.gaze
    gaze_detector = GazeDetector({"device": "CPU"})
    tracks = gaze_detector.detect_batch(frame, tracks)
    for t in tracks:
        g = t.gaze
        print(f"track {t.track_id} gaze: x={g.x:+.4f}, y={g.y:+.4f}, z={g.z:+.4f}")

    # 7) 시각화
    result = frame.copy()
    draw_tracks(result, tracks)
    draw_crop_bbox(result, tracks)
    draw_headpose(result, tracks)
    draw_gaze(result, tracks)

    # 8) 저장
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_IMAGE)
    cv2.imwrite(out_path, result)
    print(f"결과 저장 완료: {out_path}")


def image():    # LEFT_IMAGE, RIGHT_IMAGE를 직접 넣어서 계산
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) 이미지 로드
    frame = cv2.imread(INPUT_IMAGE)
    left_image = cv2.imread(LEFT_IMAGE)
    right_image = cv2.imread(RIGHT_IMAGE)
    if frame is None or left_image is None or right_image is None:
        print(f"이미지를 읽을 수 없습니다: {INPUT_IMAGE, LEFT_IMAGE, RIGHT_IMAGE}")
        return

    h, w = frame.shape[:2]
    print(f"이미지 크기(h, w): {h, w}")

    # 2) yolo detection (person)
    yolo = YoloDetector({"model": "weights/yolo/yolov8n.pt", "conf": 0.5, "classes": [0]})
    dets = yolo.detect(frame)
    if not dets:
        print("사람을 찾지 못했습니다.")
        return
    print(f"검출된 사람 수: {len(dets)}")

    # det -> track 변환
    tracks = [Track(track_id=i, bbox=d.bbox) for i, d in enumerate(dets)]

    # 3) face detection
    face_detector = FaceDetector({"device": "CPU"})
    tracks = face_detector.detect_batch(frame, tracks)

    # 4) headpose → track.headpose
    hp_estimator = HeadPoseEstimator({"device": "cpu"})
    tracks = hp_estimator.infer_batch(frame, tracks)
    for t in tracks:
        hp = t.headpose
        print(f"track {t.track_id} headpose: yaw={hp.yaw:+.1f}, pitch={hp.pitch:+.1f}, roll={hp.roll:+.1f}")

    # 5) 눈 이미지를 직접 로드하여 gaze estimation (EyeDetector 생략)
    gaze_detector = GazeDetector({"device": "CPU"})

    # 눈 이미지 전처리 (60x60, CHW, batch)
    left_resized = cv2.resize(left_image, (60, 60)).transpose((2, 0, 1))
    left_input = np.expand_dims(left_resized, axis=0)
    right_resized = cv2.resize(right_image, (60, 60)).transpose((2, 0, 1))
    right_input = np.expand_dims(right_resized, axis=0)

    for t in tracks:
        hp = t.headpose
        head_pose_angles = np.array([[hp.yaw, hp.pitch, hp.roll]], dtype=np.float32)
        infer_result = gaze_detector.compiled_model([left_input, right_input, head_pose_angles])
        results = infer_result[gaze_detector.output_layer]
        t.gaze = Gaze(x=float(results[0][0]), y=float(results[0][1]), z=float(results[0][2]))
        print(f"track {t.track_id} gaze: x={results[0][0]:+.4f}, y={results[0][1]:+.4f}, z={results[0][2]:+.4f}")

    # 5) eye detection <- gaze 벡터 시작 위치 잡기 위해서
    eye_detector = EyeDetector({"device": "CPU"})
    tracks = eye_detector.detect_batch(frame, tracks)
    for t in tracks:
        print(f"track {t.track_id} left_eye: {t.left_eye}, right_eye: {t.right_eye}")

    # 6) 시각화
    result = frame.copy()
    draw_tracks(result, tracks)
    draw_crop_bbox(result, tracks)
    draw_headpose(result, tracks)
    draw_gaze(result, tracks)

    # 7) 저장
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_IMAGE)
    cv2.imwrite(out_path, result)
    print(f"결과 저장 완료: {out_path}")


if __name__ == "__main__":
    pipeline()
    #image()
