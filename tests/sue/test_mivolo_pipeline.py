# tests/sue/test_mivolo_pipeline.py
# mivolo 테스트용
# 실행방법: PYTHONPATH=. python tests/sue/test_mivolo_pipeline.py
# 실행중지: Ctrl + C

from __future__ import annotations

from typing import Any, Dict

import cv2
from loguru import logger

from src.io.video_source import VideoSource
from src.models.yolo_detector import YoloDetector
from src.models.bytetrack_tracker import ByteTrackTracker
from src.models.face_openvino import FaceDetector
from src.models.mivolo_attr import MiVOLOAttr


def draw_attr_results(frame, tracks, attr_results):
    for track in tracks:
        bbox = track.bbox

        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2

        # 사람 bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 기본 라벨
        label = f"ID:{track.track_id}"

        # MiVOLO 결과가 있으면 뒤에 붙임
        attr = attr_results.get(track.track_id)
        if attr is not None:
            label += f" {attr.gender.name} {attr.age_group.name}"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3,
        )

        # 얼굴 bbox가 있으면 같이 그림
        if track.crop_bbox is not None:
            fb = track.crop_bbox
            cv2.rectangle(frame, (fb.x1, fb.y1), (fb.x2, fb.y2), (255, 0, 0), 3)


def main():
    # =========================
    # 경로 / 설정
    # =========================
    video_path = "data/samples/test2.mp4"
    output_path = "data/output/test_mivolo_output.mp4"

    detector_cfg: Dict[str, Any] = {
        "enabled": True,
        "model_path": "weights/yolo/yolov8n.pt",
        "device": "cpu",
        "conf_thresh": 0.25,
        "iou_thresh": 0.45,
        "classes": [0],           # 사람만
        "imgsz": 640,
    }

    tracker_cfg: Dict[str, Any] = {
        "track_thresh": 0.5,
        "low_thresh": 0.1,
        "match_thresh": 0.8,
        "max_lost_frames": 30,
        "min_hits": 1,
    }

    face_cfg: Dict[str, Any] = {
        "device": "CPU",
        "model": "weights/face_detection/face-detection-adas-0001.xml",
        "conf_thresh": 0.3,
        "min_face_size": 20,
    }

    attr_cfg: Dict[str, Any] = {
        "device": "cpu",
        "model": "weights/age_gender/model_imdb_cross_person_4.22_99.46.pth.tar",
        "repo_root": "MiVOLO",
        "min_face_size": 20,
    }

    # =========================
    # 모델 로드
    # =========================
    detector = YoloDetector(detector_cfg)
    tracker = ByteTrackTracker(tracker_cfg)
    face_detector = FaceDetector(face_cfg)
    attr_model = MiVOLOAttr(attr_cfg)

    # =========================
    # 비디오 열기
    # =========================
    vs = VideoSource(video_path)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        vs.fps,
        (vs.width, vs.height),
    )

    logger.info(f"video={video_path}")
    logger.info(f"output={output_path}")
    logger.info(f"fps={vs.fps}, size=({vs.width}, {vs.height})")

    try:
        while True:
            ok, frame, meta = vs.read()
            if not ok:
                logger.info("End of stream.")
                break

            # 1) 사람 검출
            dets = detector.detect(frame)

            # 2) tracker로 ID 부여/유지
            tracks = tracker.update(dets)

            # 3) 얼굴 검출 -> crop_bbox 채우기
            tracks = face_detector.detect_batch(frame, tracks)

            # 4) MiVOLO age/gender
            attr_results = attr_model.infer(frame, tracks)

            # 로그 확인
            logger.info(
                f"frame={meta.frame_idx} dets={len(dets)} tracks={len(tracks)} attr_results={attr_results}"
            )

            for track in tracks:
                logger.info(
                    f"[track] id={track.track_id}, bbox={track.bbox}, crop_bbox={track.crop_bbox}"
                )

            # 시각화
            draw_attr_results(frame, tracks, attr_results)

            # 저장
            writer.write(frame)

    finally:
        writer.release()
        vs.release()
        logger.info(f"Saved: {output_path}")


if __name__ == "__main__":
    main()