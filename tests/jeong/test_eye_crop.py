"""
eye_openvino.py 테스트: jpg 이미지에서
face detection -> eye detection 파이프라인을 태운 뒤
눈 영역을 crop한 이미지를 저장합니다.

사용법:
    python -m tests.jeong.test_eye_crop
"""
import cv2
import os
from src.models.face_openvino import FaceDetector
from src.models.eye_openvino import EyeDetector
from src.utils.types import BBoxXYXY, Track
from loguru import logger

INPUT_IMAGE = "data/samples/test_8.jpg"
OUTPUT_DIR = "data/output/test_eye"

OUTPUT_IMAGE = "test_8_face.jpg"
OUTPUT_LEFT = "test_8_left_eye.jpg"
OUTPUT_RIGHT = "test_8_right_eye.jpg"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) 이미지 로드
    frame = cv2.imread(INPUT_IMAGE)
    if frame is None:
        print(f"이미지를 읽을 수 없습니다: {INPUT_IMAGE}")
        return

    h, w = frame.shape[:2]
    print(f"이미지 크기(h, w): {h, w}")

    # 2) face detection
    face_detector = FaceDetector({"device": "CPU"})
    dummy_track = Track(track_id=0, bbox=BBoxXYXY(x1=0, y1=0, x2=w, y2=h))
    track = face_detector.detect(frame, dummy_track)

    if track.crop_bbox is None:
        print("얼굴을 찾지 못했습니다.")
        return

    print(f"얼굴 bbox: {track.crop_bbox}")

    # 얼굴 crop 저장
    cb = track.crop_bbox
    face_crop = frame[cb.y1:cb.y2, cb.x1:cb.x2]

    # 3) eye detection
    eye_detector = EyeDetector({"device": "CPU"})
    track= eye_detector.detect(frame, track)
    
    # 얼굴 crop 위에 랜드마크 4개 점 찍기
    face_with_pts = face_crop.copy()
    le = track.left_eye
    re = track.right_eye
    for (px, py) in [(le.x1, le.y1), (le.x2, le.y2)]:
        cv2.circle(face_with_pts, (int(px), int(py)), 3, (0, 0, 255), -1)
    for (px, py) in [(re.x1, re.y1), (re.x2, re.y2)]:
        cv2.circle(face_with_pts, (int(px), int(py)), 3, (0, 255, 0), -1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, OUTPUT_IMAGE), face_with_pts)

    if track.left_eye is None or track.right_eye is None:
        print("눈을 찾지 못했습니다.")
        return

    print(f"왼쪽 눈 bbox: {track.left_eye}")
    print(f"오른쪽 눈 bbox: {track.right_eye}")

    # 4) 눈 영역 crop 저장
    le = track.left_eye
    re = track.right_eye
    left_crop = frame[le.y1:le.y2, le.x1:le.x2]
    right_crop = frame[re.y1:re.y2, re.x1:re.x2]

    cv2.imwrite(os.path.join(OUTPUT_DIR, OUTPUT_LEFT), left_crop)
    cv2.imwrite(os.path.join(OUTPUT_DIR, OUTPUT_RIGHT), right_crop)
    print(f"결과 저장 완료: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
