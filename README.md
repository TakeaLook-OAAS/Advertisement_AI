# Advertisement AI — 광고 시청 분석 시스템

CCTV/카메라 영상에서 **사람 검출 → 추적 → 얼굴 검출 → 머리 방향 추정 → 광고 시청 판정**까지 자동으로 수행하여, 광고판 앞 유동·체류·관심 인구를 분석하는 파이프라인입니다.

## 파이프라인 개요

```
영상 입력 (MP4 / 웹캠)
  │
  ▼  매 프레임마다 Orchestrator.process(frame, meta)
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  1) YOLO 사람 검출        yolo_detector.py                     │
│     frame → List[Det]    (person bbox + confidence)            │
│         │                                                      │
│  2) ByteTrack 추적       bytetrack_tracker.py                  │
│     List[Det] → List[Track]  (track_id + bbox 유지)            │
│         │                                                      │
│  3) 얼굴 검출 (OpenVINO)  face_openvino.py                     │
│     person bbox crop → face detection → Track.crop_bbox 갱신   │
│         │                                                      │
│  4) 머리 방향 추정        headpose_6drepnet.py                  │
│     crop_bbox crop → 6DRepNet → yaw/pitch/roll                 │
│         │                                                      │
│  5) 시선 추정 (TODO)      gaze_openvino.py                     │
│     양쪽 눈 + headpose → 3D 시선 벡터                           │
│         │                                                      │
│  6) 광고 시청 판정         logic/attention.py, look_judge.py    │
│     yaw/pitch 임계값 또는 시선 벡터 코사인 유사도 기반 판정       │
│         │                                                      │
│  7) ROI/체류 판정          logic/roi.py, stay.py                │
│     bbox 중심점이 ROI 폴리곤 안에 있는지 + 체류 시간 계산         │
│         │                                                      │
│  8) 이벤트/통계 기록       logic/status.py                      │
│     이벤트 기반 로깅 (JSONL) → 일/주/월 집계                     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
출력: 시각화 비디오 (MP4) + 이벤트 로그 + 통계
```

## 프로젝트 구조

```
Advertisement_AI/
├── configs/
│   └── dev.yaml              # 개발 환경 설정 (소스, 모델 경로, 임계값 등)
├── data/
│   ├── samples/              # 입력 영상
│   └── output/               # 출력 영상
├── weights/                  # 모델 가중치 파일 (git 미추적)
│   ├── yolo/                 #   YOLOv8 (.pt)
│   ├── face_detection/       #   OpenVINO face-detection (.xml/.bin)
│   └── headpose/             #   6DRepNet (.pth)
├── src/
│   ├── main.py               # 진입점
│   ├── pipeline/
│   │   ├── orchestrator.py   # 컨트롤 타워 (모든 모델/로직 호출)
│   │   └── runner.py         # 영상 루프 + 시각화 + 비디오 저장
│   ├── models/
│   │   ├── yolo_detector.py      # YOLOv8 사람 검출
│   │   ├── bytetrack_tracker.py  # ByteTrack 다중 객체 추적
│   │   ├── face_openvino.py      # OpenVINO 얼굴 검출
│   │   ├── headpose_6drepnet.py  # 6DRepNet 머리 방향 추정
│   │   ├── gaze_openvino.py      # OpenVINO 시선 추정
│   │   └── mivolo_attr.py        # MiVOLO 나이/성별 추정
│   ├── logic/
│   │   ├── attention.py      # yaw/pitch 기반 광고 시청 판정
│   │   ├── look_judge.py     # 시선 벡터 기반 정밀 판정
│   │   ├── roi.py            # 다각형 ROI 내부 판정
│   │   ├── stay.py           # 체류 시간 계산 로직
│   │   └── status.py         # 이벤트 기반 통계 집계
│   ├── io/
│   │   └── video_source.py   # OpenCV VideoCapture 래퍼
│   ├── vision/
│   │   └── draw.py           # 시각화 (bbox, headpose, FPS 드로잉)
│   └── utils/
│       ├── types.py          # 공용 데이터 타입 (Det, Track, HeadPose 등)
│       └── config.py         # YAML 설정 로드
└── docker/
    └── Dockerfile            # Docker 빌드 설정
```

## 핵심 데이터 흐름

각 단계의 **입력 → 출력** 데이터 타입:

| 단계 | 모듈 | 입력 | 출력 |
|---|---|---|---|
| 1. 사람 검출 | `yolo_detector.py` | `frame (np.ndarray)` | `List[Det]` (bbox + conf) |
| 2. 추적 | `bytetrack_tracker.py` | `List[Det]` | `List[Track]` (track_id + bbox) |
| 3. 얼굴 검출 | `face_openvino.py` | `frame, List[Track]` | `List[Track]` (crop_bbox 갱신) |
| 4. 머리 방향 | `headpose_6drepnet.py` | `frame, List[Track]` | `List[(track_id, HeadPose, reason)]` |
| 5. 시선 추정 | `gaze_openvino.py` | `frame, crop_bbox, HeadPose` | `Gaze` (3D 벡터) |
| 6. 시청 판정 | `attention.py` | `HeadPose` | `bool` (보고 있는지) |
| 7. 체류 판정 | `roi.py` + `stay.py` | `Track, ROI` | `in_roi, dwell_frames` |
| 8. 통계 기록 | `status.py` | `Event` | JSONL 로그 |

## Track의 bbox vs crop_bbox

```
Track.bbox      = YOLO가 검출한 사람 전체 영역 (원본 유지)
Track.crop_bbox = FaceDetector가 찾은 얼굴 영역 (Optional, 없으면 None)
                  → HeadPose에서 사용. None이면 bbox로 대체
```

## 사용되는 모델

| 모델 | 프레임워크 | 용도 | 파일 |
|---|---|---|---|
| YOLOv8n | PyTorch (ultralytics) | 사람 검출 | `yolov8n.pt` |
| ByteTrack | 순수 Python | 다중 객체 추적 | 코드 내장 |
| face-detection-adas-0001 | OpenVINO | 얼굴 검출 | `.xml` + `.bin` |
| 6DRepNet | PyTorch (sixdrepnet) | 머리 방향 추정 | `.pth` |
| gaze-estimation | OpenVINO | 시선 추정 | `.xml` + `.bin` |
| MiVOLO | PyTorch | 나이/성별 추정 | `.pt` |

## 실행 방법

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 설정 파일 확인/수정
#    configs/dev.yaml에서 source, model 경로 등 설정

# 3. 실행
python -m src.main
```

## 설정 파일 (dev.yaml) <- 2026-03-05 기준

```yaml
source: "data/samples/test2.mp4"     # 입력 소스 (파일 경로 또는 웹캠 번호)

display:
  output_video: true                  # MP4 출력 여부
  output_video_path: "data/output/output.mp4"
  draw_bbox: true                     # bbox 시각화
  draw_headpose: true                 # headpose 텍스트
  draw_headpose_vector: true          # headpose 3D 축

models:
  yolo:
    model: "weights/yolo/yolov8n.pt"
    conf: 0.5
    classes: [0]                      # 0=person
  face:
    model: "weights/face_detection/face-detection-adas-0001.xml"
    conf_thresh: 0.5
  headpose:
    weights: "weights/headpose/6DRepNet_300W_LP_AFLW2000.pth"
    min_face_size: 30
```

## 최종 목표 출력

- **시각화 비디오**: bbox + track ID + headpose 방향 표시
- **이벤트 로그** (JSONL): 유동(pass_by), 진입/이탈(enter/exit_roi), 체류(dwell), 시청(look) 이벤트
- **통계 집계**: 유동 인구, 체류 인구/시간, 관심 인구/시간, 체류율, 관심률
