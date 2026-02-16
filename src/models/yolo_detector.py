# detect(frame)->List[Det]

from __future__ import annotations
from typing import Any, Dict, List
from src.utils.types import Det


class YoloDetector:
    """
    YOLO 어댑터:
      detect(frame) -> List[Det]

    지금은 stub(빈 리스트 반환).
    나중에 ultralytics/onnx/openvino 등으로 교체하더라도
    이 인터페이스만 유지하면 파이프라인이 안 깨짐.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def detect(self, frame) -> List[Det]:
        # TODO: 실제 YOLO 붙이면 여기서 raw -> Det로 정규화해서 반환
        return []