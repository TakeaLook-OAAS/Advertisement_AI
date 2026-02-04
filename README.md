# AI Module Starter (OpenCV + 6DRepNet + OpenVINO Gaze)

This is a **starter folder structure** for the AI part of your graduation project:

- OpenCV: video input, frame loop, FPS measurement, visualization
- 6DRepNet (PyTorch): head pose estimation (yaw/pitch/roll)
- OpenVINO: gaze vector inference
- Docker: consistent environment for multiple AI teammates

> This repo is a **runnable skeleton**: models are stubbed until you drop in your weights/IR files.

## Quick start (Docker)

1) Build & run (webcam by default):
```bash
cd ai
docker compose up --build
```

2) Run with a video file:
```bash
docker compose run --rm ai python -m src.main --config configs/dev.yaml --source data/samples/test.mp4
```

3) Run without showing a window (server/headless):
```bash
docker compose run --rm ai python -m src.main --config configs/dev.yaml --source 0 --no-window
```

## Where to put model files

- 6DRepNet weights:
  - `models/headpose/weights/` (ignored by git by default)
- OpenVINO IR files:
  - `models/gaze/ir/` (ignored by git by default)

## Notes
- If you want to use RTSP: set `--source rtsp://...` or edit `configs/*.yaml`.
- Press `q` to quit when window mode is enabled.

