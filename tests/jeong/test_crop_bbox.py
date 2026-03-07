"""
crop_bbox 시각화 테스트: 영상에서 YOLO → ByteTrack → FaceDetector 파이프라인을 태운 뒤
person bbox + crop_bbox(얼굴 영역)를 그린 영상을 출력합니다.

사용법:
    python -m tests.jeong.test_crop_bbox
"""
import cv2
import os
from src.models.yolo_detector import YoloDetector
from src.models.bytetrack_tracker import ByteTrackTracker
from src.models.face_openvino import FaceDetector
from src.vision.draw import draw_tracks, draw_crop_bbox

INPUT_VIDEO = "data/samples/test2.mp4"
OUTPUT_VIDEO = "data/output/test_output_cropbbox.mp4"


def main():
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"영상을 열 수 없습니다: {INPUT_VIDEO}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    yolo = YoloDetector({"device": "cpu", "model_path": "weights/yolo/yolov8n.pt", "classes": [0]})
    tracker = ByteTrackTracker({})
    face = FaceDetector({"device": "CPU"})

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) YOLO detection → tracking
        dets = yolo.detect(frame)
        tracks = tracker.update(dets)

        # 2) face detection → crop_bbox 갱신
        tracks = face.detect_batch(frame, tracks)

        # 3) 시각화
        draw_tracks(frame, tracks)
        draw_crop_bbox(frame, tracks)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"[{frame_idx}/{total}] 처리 중...")

    cap.release()
    writer.release()
    print(f"완료: {OUTPUT_VIDEO} ({frame_idx} frames)")


if __name__ == "__main__":
    main()
