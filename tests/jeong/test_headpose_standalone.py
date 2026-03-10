# HeadPose 독립 테스트 스크립트
# YOLO → ByteTrack → FaceDetector(OpenVINO) → HeadPose(6DRepNet) 파이프라인 테스트
#
# 실행 방법 (컨테이너 터미널에서):
#   python -m tests.jeong.test_headpose_standalone --source data/samples/test.jpg
#   python -m tests.jeong.test_headpose_standalone --source data/samples/test.jpg --output data/output/headpose_result.jpg

import argparse
import os
import cv2
from loguru import logger

from src.models.yolo_detector import YoloDetector
from src.models.bytetrack_tracker import ByteTrackTracker
from src.models.face_openvino import FaceDetector
from src.models.headpose_6drepnet import HeadPoseEstimator
from src.vision.draw import draw_tracks, draw_crop_bbox, draw_headpose


def parse_args():
    p = argparse.ArgumentParser(description="HeadPose 파이프라인 테스트 (이미지)")
    p.add_argument("--source", type=str, default="data/samples/test.jpg")
    p.add_argument("--output", type=str, default="data/output/test_0_headpose.jpg")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main():
    args = parse_args()

    # 이미지 읽기
    frame = cv2.imread(args.source)
    if frame is None:
        logger.error(f"이미지를 열 수 없습니다: {args.source}")
        return

    logger.info(f"입력 이미지: {args.source} ({frame.shape[1]}x{frame.shape[0]})")

    # 1) YOLO 검출
    yolo = YoloDetector({"device": args.device})
    dets = yolo.detect(frame)
    logger.info(f"[YOLO] 검출 수: {len(dets)}")
    for i, d in enumerate(dets):
        logger.info(f"  det[{i}] bbox=({d.bbox.x1},{d.bbox.y1},{d.bbox.x2},{d.bbox.y2}) conf={d.conf:.3f}")

    if not dets:
        logger.warning("검출된 객체가 없습니다.")
        return

    # 2) ByteTrack 트래킹
    tracker = ByteTrackTracker({})
    tracks = tracker.update(dets)
    logger.info(f"[ByteTrack] 트랙 수: {len(tracks)}")
    for t in tracks:
        logger.info(f"  track_id={t.track_id} bbox=({t.bbox.x1},{t.bbox.y1},{t.bbox.x2},{t.bbox.y2})")

    if not tracks:
        logger.warning("트래킹된 객체가 없습니다.")
        return

    # 3) Face Detection (OpenVINO)
    face_detector = FaceDetector({"device": "CPU"})
    tracks = face_detector.detect_batch(frame, tracks)
    logger.info(f"[FaceDetector] 얼굴 검출 완료")
    for t in tracks:
        if t.crop_bbox is not None:
            logger.info(f"  track_id={t.track_id} crop_bbox=({t.crop_bbox.x1},{t.crop_bbox.y1},{t.crop_bbox.x2},{t.crop_bbox.y2})")
        else:
            logger.info(f"  track_id={t.track_id} crop_bbox=None")

    # 4) HeadPose 추정 (6DRepNet) → track.headpose
    hp_estimator = HeadPoseEstimator({"device": args.device})
    tracks = hp_estimator.infer_batch(frame, tracks)
    logger.info(f"[HeadPose] 추정 완료")
    for t in tracks:
        hp = t.headpose
        logger.info(f"  track_id={t.track_id} yaw={hp.yaw:+.1f} pitch={hp.pitch:+.1f} roll={hp.roll:+.1f}")

    # 5) 시각화 (draw.py 활용)
    draw_tracks(frame, tracks)
    draw_crop_bbox(frame, tracks)
    draw_headpose(frame, tracks)

    # 결과 저장
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, frame)
    logger.info(f"결과 저장: {args.output}")


if __name__ == "__main__":
    main()
