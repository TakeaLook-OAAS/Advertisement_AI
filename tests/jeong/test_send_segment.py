"""
기존 segment JSON 파일을 백엔드로 전송하는 테스트 스크립트.

사용법:
  # 기본 (segment_000.json → localhost:8000)
  python tests/jeong/test_send_segment.py

  # 파일 지정
  python tests/jeong/test_send_segment.py --file data/output/segments/segment_001.json

  # 폴더 안 전체 전송
  python tests/jeong/test_send_segment.py --dir data/output/segments/

  # device_id 덮어쓰기 (DB에 등록된 UUID로)
  python tests/jeong/test_send_segment.py --device-id xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

  # 백엔드 URL 변경
  python tests/jeong/test_send_segment.py --url http://192.168.0.10:8000/api/v1/events/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

# ── 기본값 ──────────────────────────────────────────────────────────────────
DEFAULT_URL = "http://back_dev:8000/events/"
DEFAULT_FILE = "data/output/segments/segment_000.json"


def send(data: dict, url: str) -> None:
    """JSON dict 하나를 백엔드로 전송하고 결과를 출력한다."""
    seg = data.get("segment", {})
    label = f"index={seg.get('index')} cycle={seg.get('cycle_index')} device={seg.get('device_id')}"

    try:
        resp = requests.post(url, json=data, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] 네트워크 오류: {e}")
        return

    if resp.status_code == 202:
        inserted = resp.json().get("inserted", "?")
        print(f"  [OK] {label} → inserted={inserted}")
    else:
        print(f"  [FAIL] {label} → status={resp.status_code} body={resp.text[:200]}")


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment JSON → Backend 전송 테스트")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", default=DEFAULT_FILE, help="전송할 JSON 파일 경로")
    group.add_argument("--dir", help="폴더 안의 모든 segment_*.json 전송")
    parser.add_argument("--url", default=DEFAULT_URL, help="백엔드 엔드포인트 URL")
    parser.add_argument("--device-id", help="segment.device_id 덮어쓰기 (미등록 기기 테스트 우회)")
    args = parser.parse_args()

    # ── 전송할 파일 목록 결정 ────────────────────────────────────────────────
    if args.dir:
        files = sorted(Path(args.dir).glob("segment_*.json"))
        if not files:
            print(f"[ERROR] {args.dir} 에 segment_*.json 파일이 없습니다.")
            sys.exit(1)
    else:
        files = [Path(args.file)]
        if not files[0].exists():
            print(f"[ERROR] 파일을 찾을 수 없습니다: {files[0]}")
            sys.exit(1)

    print(f"URL: {args.url}")
    print(f"파일 {len(files)}개 전송 시작...\n")

    for path in files:
        data = load_json(path)
        if args.device_id:
            data["segment"]["device_id"] = args.device_id
        print(f"파일: {path.name}")
        send(data, args.url)

    print("\n완료.")


if __name__ == "__main__":
    main()
