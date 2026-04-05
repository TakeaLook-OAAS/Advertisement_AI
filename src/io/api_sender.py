from __future__ import annotations

import json
from typing import Any, Dict

import requests
from loguru import logger


def send_segment(segment_data: Dict[str, Any], url: str) -> None:
    """
    세그먼트 JSON을 백엔드 POST /events/ 로 전송한다.
    실패해도 예외를 올리지 않고 경고 로그만 남긴다.
    """
    try:
        resp = requests.post(url, json=segment_data, timeout=10)
        if resp.status_code == 202:
            inserted = resp.json().get("inserted", "?")
            logger.info(f"Backend 전송 성공: inserted={inserted} | url={url}")
        else:
            logger.warning(
                f"Backend 전송 실패: status={resp.status_code} | body={resp.text[:200]}"
            )
    except requests.exceptions.RequestException as e:
        logger.warning(f"Backend 전송 오류 (네트워크): {e}")
