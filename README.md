# Advertisement AI — 광고 시청 분석 시스템

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenVINO](https://img.shields.io/badge/OpenVINO-2024-0071C5?style=for-the-badge&logo=intel&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

> CCTV/카메라 영상에서 **사람 검출 → 추적 → 얼굴 검출 → 나이/성별 추정 → 머리 방향 추정 → 시선 추정 → 광고 시청 판정**까지 자동으로 수행하여, 광고판 앞 유동·체류·관심 인구를 분석하고 백엔드로 전송하는 AI 파이프라인

---

## 파이프라인 개요

```
영상 입력 (MP4 / 웹캠)
  │
  ▼  매 프레임마다 Orchestrator.process(frame)
──────────────────────────────────────────────────────────────
  1) YOLO 사람 검출           yolo_detector.py
     frame → List[Det]       (person bbox + confidence)
         │
  2) ByteTrack 추적           bytetrack_tracker.py
     List[Det] → List[Track] (track_id + bbox 유지)
         │
  3) 얼굴 검출 (OpenVINO)     face_openvino.py
     person bbox → face bbox → Track.crop_bbox 갱신
         │
  4) 나이/성별 추정 (MiVOLO)  mivolo_attr.py
     person + face crop → age_group / gender
         │
  5) 머리 방향 추정            headpose_6drepnet.py
     face crop → 6DRepNet → yaw / pitch / roll
         │
  6) 눈 검출 (OpenVINO)       eye_openvino.py
     face crop → left_eye / right_eye bbox
         │
  7) 시선 추정 (OpenVINO)     gaze_openvino.py
     눈 crop + headpose → 3D 시선 벡터
         │
  8) 광고 시청 판정             look_judge.py
     yaw/pitch 임계값 + 시선 벡터 기반 최종 판정
         │
  9) ROI/체류 판정              logic/stay.py
     bbox 중심점이 ROI 폴리곤 안에 있는지 + 체류 시간 계산
         │
 10) 상태 추적 및 세그먼트 기록  logic/status.py
     광고 사이클 단위로 통계 집계 → JSON 저장 → 백엔드 전송
──────────────────────────────────────────────────────────────
  │
  ▼
출력: 세그먼트 JSON + 시각화 비디오 (MP4) + 백엔드 자동 전송
```

---

## 프로젝트 구조

```
Advertisement_AI/
├── configs/
│   └── dev.yaml                  # 환경 설정 (소스, 모델 경로, 임계값, 백엔드 URL 등)
├── data/
│   ├── samples/                  # 입력 영상
│   └── output/
│       └── segments/             # 세그먼트 JSON 출력
├── weights/                      # 모델 가중치 파일 (git 미추적)
│   ├── yolo/                     #   YOLOv8 (.pt)
│   ├── face_detection/           #   OpenVINO face-detection (.xml/.bin)
│   ├── headpose/                 #   6DRepNet (.pth)
│   ├── eye_detection/            #   OpenVINO facial-landmarks (.xml/.bin)
│   ├── gaze/                     #   OpenVINO gaze-estimation (.xml/.bin)
│   └── age_gender/               #   MiVOLO (.pth)
├── src/
│   ├── main.py                   # 진입점
│   ├── pipeline/
│   │   ├── orchestrator.py       # 컨트롤 타워 (모든 모델/로직 조율)
│   │   └── runner.py             # 영상 루프 + 광고 사이클 관리 + 비디오 저장
│   ├── models/
│   │   ├── yolo_detector.py      # YOLOv8 사람 검출
│   │   ├── bytetrack_tracker.py  # ByteTrack 다중 객체 추적
│   │   ├── face_openvino.py      # OpenVINO 얼굴 검출
│   │   ├── mivolo_attr.py        # MiVOLO 나이/성별 추정
│   │   ├── headpose_6drepnet.py  # 6DRepNet 머리 방향 추정
│   │   ├── eye_openvino.py       # OpenVINO 눈 검출
│   │   └── gaze_openvino.py      # OpenVINO 시선 추정
│   ├── logic/
│   │   ├── look_judge.py         # 시선 벡터 + headpose 기반 최종 시청 판정
│   │   ├── stay.py               # ROI 진입/이탈 및 체류 시간 계산
│   │   ├── status.py             # 트랙별 이벤트 상태 추적 + 세그먼트 집계
│   │   └── ad_cycle.py           # 광고 사이클 스케줄러 (YAML durations_s 기반)
│   ├── io/
│   │   ├── video_source.py       # OpenCV VideoCapture 래퍼
│   │   └── api_sender.py         # 세그먼트 JSON → 백엔드 POST 전송
│   ├── vision/
│   │   └── draw.py               # 시각화 (bbox, headpose, gaze, look, gender/age)
│   └── utils/
│       ├── types.py              # 공용 데이터 타입 (Det, Track, HeadPose, AdSegmentInfo 등)
│       └── config.py             # YAML 설정 로드
├── docker/
│   ├── Dockerfile                # CPU 빌드
│   ├── Dockerfile.gpu            # CUDA 12.1 + PyTorch GPU 빌드
│   ├── docker-compose.yml        # 컨테이너 구성
│   └── patch_mivolo.py           # timm 1.0.3 호환 패치
├── tests/                        # 기능별 테스트 스크립트
├── MiVOLO/                       # MiVOLO 서브모듈
├── pyproject.toml
├── requirements.txt
└── .env.example
```

---

## 사용 모델

| 모델 | 프레임워크 | 용도 | 가중치 파일 |
|---|---|---|---|
| YOLOv8n | PyTorch (ultralytics) | 사람 검출 | `weights/yolo/yolov8n.pt` |
| ByteTrack | 순수 Python | 다중 객체 추적 | 코드 내장 |
| face-detection-adas-0001 | OpenVINO | 얼굴 검출 | `weights/face_detection/*.xml/.bin` |
| MiVOLO | PyTorch | 나이/성별 추정 | `weights/age_gender/*.pth` |
| 6DRepNet | PyTorch (sixdrepnet) | 머리 방향 추정 | `weights/headpose/*.pth` |
| facial-landmarks-35-adas-0002 | OpenVINO | 눈 검출 | `weights/eye_detection/*.xml/.bin` |
| gaze-estimation-adas-0002 | OpenVINO | 시선 추정 | `weights/gaze/*.xml/.bin` |

---

## 실행 방법

```bash
# 1. 레포 클론
git clone --recurse-submodules https://github.com/TakeaLook-OAAS/Advertisement_AI.git
cd Advertisement_AI

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 가중치 파일을 weights/ 하위 경로에 배치

# 4. 설정 파일 수정
#    configs/dev.yaml 에서 source, device_id, 모델 경로, 백엔드 URL 등 설정

# 5. 실행
python -m src.main --config configs/dev.yaml
```

Docker를 사용하는 경우 `docker/docker-compose.yml`을 참고. CPU용(`Dockerfile`)과 GPU용(`Dockerfile.gpu`) 이미지 모두 존재함.

---

## 설정 파일 (configs/dev.yaml)

```yaml
device_id: "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"  # 백엔드 등록 기기 UUID
source: "data/samples/test.mp4"                     # 입력 소스 (파일 경로 또는 웹캠 번호)

display:
  output_video: false
  output_video_path: "data/output/output.mp4"
  draw_bbox: true
  draw_crop_bbox: true
  draw_fps: true
  draw_headpose: true
  draw_gaze: true
  draw_roi: true
  draw_look: true
  draw_gender_age: true

pipeline:
  frame_skip: 10        # N프레임마다 1번 처리 (1=매프레임, 10=10프레임마다)

models:
  yolo:
    device: "cuda"      # cpu / cuda
    model: "weights/yolo/yolov8n.pt"
    conf: 0.5
    classes: [0]        # 0=person
  tracker:
    iou_threshold: 0.3
    max_lost: 30
  face:
    device: "CPU"
    weights: "weights/face_detection/face-detection-adas-0001.xml"
    conf_thresh: 0.5
  mivolo:
    device: "cuda"
    model: "weights/age_gender/model_imdb_cross_person_4.22_99.46.pth"
  headpose:
    device: "cuda"
    weights: "weights/headpose/6DRepNet_300W_LP_AFLW2000.pth"
  eye:
    device: "CPU"
    weights: "weights/eye_detection/facial-landmarks-35-adas-0002.xml"
  gaze:
    device: "CPU"
    weights: "weights/gaze/gaze-estimation-adas-0002.xml"

logic:
  attention:
    threshold_deg: 30.0

output:
  json_dir: "data/output/segments/"
  ad_cycle:
    durations_s: [30, 25, 35]  # 광고 사이클별 재생 시간(초) 목록

backend:
  url: "http://back_dev:8000/events/"  # 백엔드 수신 엔드포인트
```
