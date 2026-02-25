# HeadPose 독립 테스트 스크립트
# YOLO / ByteTrack 없이 HeadPose + Attention 판정만 테스트
#
# 실행 방법 (컨테이너 터미널에서):
#   python -m tests.jeong.test_headpose_standalone --source data/samples/test.mp4
#   python -m tests.jeong.test_headpose_standalone --source data/samples/test.mp4 --output data/output/result.mp4

import argparse
import os
import math
import cv2
from loguru import logger

from src.models.headpose_6drepnet import HeadPoseEstimator
from src.logic.attention import is_attending
from src.utils.types import BBoxXYXY, Track


def parse_args():
    p = argparse.ArgumentParser(description="HeadPose 독립 테스트 (YOLO/ByteTrack 없음)")
    p.add_argument("--source", type=str, default="data/samples/test.mp4")
    p.add_argument("--output", type=str, default="data/output/headpose_test.mp4")  # 결과 저장 경로
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max-yaw", type=float, default=25)
    p.add_argument("--max-pitch", type=float, default=20)
    return p.parse_args()


def main():
    args = parse_args()

    # 소스 열기
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"영상을 열 수 없습니다: {source}")
        return

    # HeadPose 모델 로드
    hp_estimator = HeadPoseEstimator({"device": args.device, "min_face_size": 30})
    logger.info("HeadPose 모델 로드 완료")

    # Attention 판정 설정
    attention_cfg = {
        "max_yaw_abs_deg": args.max_yaw,
        "max_pitch_abs_deg": args.max_pitch,
    }

    # VideoWriter 준비
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w_frame, h_frame),
    )
    logger.info(f"결과 저장 경로: {args.output} ({w_frame}x{h_frame} @ {fps:.1f}fps)")

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            logger.info("영상 끝")
            break

        h, w = frame.shape[:2]

        # 가짜 bbox: 프레임 전체를 하나의 사람 영역으로 간주
        dummy_bbox = BBoxXYXY(x1=0, y1=0, x2=w, y2=h)
        dummy_track = Track(track_id=1, bbox=dummy_bbox)

        # HeadPose 추정
        hp, fail_reason = hp_estimator.infer(frame, dummy_track)

        if hp is not None:
            # Attention 판정
            hp_dict = {"yaw": hp.yaw, "pitch": hp.pitch}
            attending = is_attending(hp_dict, attention_cfg)

            # 텍스트 표시
            color = (0, 255, 0) if attending else (0, 0, 255)
            status = "ATTENDING" if attending else "NOT ATTENDING"

            cv2.putText(frame, f"Yaw: {hp.yaw:+.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # 좌측 상단에 Yaw 값
            cv2.putText(frame, f"Pitch: {hp.pitch:+.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # 그 아래 Pitch 값
            cv2.putText(frame, f"Roll: {hp.roll:+.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # 그 아래 Roll 값
            cv2.putText(frame, status, (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)            # 그 아래 ATTENDING/NOT ATTENDING

            # 방향 화살표
            cx, cy = w // 2, h // 2     # 화면 중앙 좌표
            arrow_len = min(w, h) // 4  # 화살표 길이 (화면 크기의 1/4)
            dx = int(arrow_len * math.sin(math.radians(hp.yaw)))    # yaw → 좌우(X) 이동량
            dy = int(-arrow_len * math.sin(math.radians(hp.pitch))) # pitch → 상하(Y) 이동량
            cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy), color, 3, tipLength=0.3)

        else:
            cv2.putText(frame, f"HeadPose FAIL: {fail_reason}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 프레임 번호
        cv2.putText(frame, f"Frame: {frame_idx}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        writer.write(frame)  # 프레임을 파일에 저장
        frame_idx += 1

    writer.release()
    cap.release()
    logger.info(f"완료: {frame_idx}프레임 저장 → {args.output}")


if __name__ == "__main__":
    main()
