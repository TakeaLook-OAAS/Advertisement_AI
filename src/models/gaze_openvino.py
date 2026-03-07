# infer(left_eye, right_eye, headpose)->Gaze

from __future__ import annotations
from typing import Any, Dict, List
import cv2
import numpy as np
from loguru import logger
from openvino import Core
from src.utils.types import BBoxXYXY, Track










































# https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/gaze-estimation-adas-0002/FP32/