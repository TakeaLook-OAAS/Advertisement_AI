# src/models/mivolo_attr.py
# infer(frame, tracks) -> Dict[track_id, PersonAttr]

from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import sys

import numpy as np
from loguru import logger

from src.utils.types import (
    AgeGroup,
    AttrMap,
    Gender,
    PersonAttr,
    Track,
)


class MiVOLOAttr:
    """
    MiVOLO attribute inference wrapper

    input:
        frame: np.ndarray
        tracks: List[Track]

    output:
        Dict[track_id, PersonAttr]
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.device = cfg.get("device", "cpu")
        self.weights = cfg.get(
            "model",
            "weights/age_gender/model_imdb_cross_person_4.22_99.46.pth.tar",
        )
        self.repo_root = cfg.get("repo_root", "/app/MiVOLO")
        self.min_face_size = int(cfg.get("min_face_size", 20))
        self.min_person_size = int(cfg.get("min_person_size", 40))
        self.use_persons = bool(cfg.get("use_persons", True))

        self.model = self._load_model()

        # 체크포인트 메타와 실제 설정 동기화
        if hasattr(self.model, "meta") and hasattr(self.model.meta, "with_persons_model"):
            self.use_persons = bool(self.model.meta.with_persons_model)

        logger.info(
            f"[MiVOLOAttr] weights={self.weights} "
            f"device={self.device} repo_root={self.repo_root} "
            f"use_persons={self.use_persons}"
        )

    def _load_model(self):
        """
        MiVOLO 모델 직접 로드
        """
        repo_abs = os.path.abspath(self.repo_root)
        if repo_abs not in sys.path:
            sys.path.append(repo_abs)

        try:
            from mivolo.model.mi_volo import MiVOLO
        except ImportError as e:
            raise ImportError(
                "MiVOLO model import failed. "
                f"Check repo_root={repo_abs}"
            ) from e

        use_half = str(self.device).lower().startswith("cuda")

        model = MiVOLO(
            ckpt_path=self.weights,
            device=self.device,
            half=use_half,
            disable_faces=False,
            use_persons=self.use_persons,
            verbose=False,
        )
        return model
    
    def infer(self, frame: np.ndarray, tracks: List[Track]) -> List[Track]:
        """
        각 track의 attr 필드를 채워서 반환
        """
        for track in tracks:
            attr = self._infer_one(frame, track)
            track.attr = attr

        return tracks

    def _infer_one(self, frame: np.ndarray, track: Track) -> Optional[PersonAttr]:
        logger.info(
            f"[MiVOLOAttr] track_id={track.track_id}, "
            f"person_bbox={track.bbox}, face_bbox={track.crop_bbox}"
        )

        face_img = self._crop_face(frame, track)
        if face_img is None:
            logger.warning(f"[MiVOLOAttr] no valid face crop for track_id={track.track_id}")
            return None

        person_img = None
        if self.use_persons:
            person_img = self._crop_person(frame, track)
            if person_img is None:
                logger.warning(f"[MiVOLOAttr] no valid person crop for track_id={track.track_id}")
                return None

        try:
            pred = self._predict(face_img, person_img)
        except Exception:
            logger.exception(
                f"[MiVOLOAttr] prediction failed for track_id={track.track_id}"
            )
            return None

        if pred is None:
            return None

        age, gender_value = pred

        logger.info(
            f"[MiVOLOAttr] parsed pred for track_id={track.track_id}: "
            f"age={age}, gender={gender_value}"
        )

        return PersonAttr(
            gender=self._to_gender(gender_value),
            age_group=self._to_age_group(age),
        )

    def _crop_face(self, frame: np.ndarray, track: Track) -> Optional[np.ndarray]:
        """
        track.crop_bbox를 사용해 얼굴 crop
        """
        bbox = track.crop_bbox
        if bbox is None:
            return None

        h, w = frame.shape[:2]

        x1 = max(0, min(int(bbox.x1), w - 1))
        y1 = max(0, min(int(bbox.y1), h - 1))
        x2 = max(0, min(int(bbox.x2), w))
        y2 = max(0, min(int(bbox.y2), h))

        if x2 <= x1 or y2 <= y1:
            return None

        if (x2 - x1) < self.min_face_size or (y2 - y1) < self.min_face_size:
            return None

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None

        return face

    def _crop_person(self, frame: np.ndarray, track: Track) -> Optional[np.ndarray]:
        """
        track.bbox를 사용해 사람 crop
        """
        bbox = track.bbox
        if bbox is None:
            return None

        h, w = frame.shape[:2]

        x1 = max(0, min(int(bbox.x1), w - 1))
        y1 = max(0, min(int(bbox.y1), h - 1))
        x2 = max(0, min(int(bbox.x2), w))
        y2 = max(0, min(int(bbox.y2), h))

        if x2 <= x1 or y2 <= y1:
            return None

        if (x2 - x1) < self.min_person_size or (y2 - y1) < self.min_person_size:
            return None

        person = frame[y1:y2, x1:x2]
        if person.size == 0:
            return None

        return person

    def _predict(
        self,
        face_img: np.ndarray,
        person_img: Optional[np.ndarray] = None,
    ):
        """
        얼굴 crop 하나(or 얼굴+사람 crop)를 MiVOLO 모델에 직접 넣어 age/gender 추론
        """
        from mivolo.data.misc import prepare_classification_images

        faces_input = prepare_classification_images(
            [face_img],
            self.model.input_size,
            self.model.data_config["mean"],
            self.model.data_config["std"],
            device=self.model.device,
        )

        if faces_input is None:
            raise ValueError("prepare_classification_images returned None for face")

        if self.use_persons:
            if person_img is None:
                raise ValueError("person_img is required when use_persons=True")

            persons_input = prepare_classification_images(
                [person_img],
                self.model.input_size,
                self.model.data_config["mean"],
                self.model.data_config["std"],
                device=self.model.device,
            )

            if persons_input is None:
                raise ValueError("prepare_classification_images returned None for person")

            # face + person => [B, 6, H, W]
            model_input = np.concatenate(
                [faces_input.cpu().numpy(), persons_input.cpu().numpy()],
                axis=1,
            )

            import torch
            model_input = torch.from_numpy(model_input).to(self.model.device)

            logger.info(
                f"[MiVOLOAttr] face+person input shape={tuple(model_input.shape)}"
            )
        else:
            model_input = faces_input
            logger.info(
                f"[MiVOLOAttr] face-only input shape={tuple(model_input.shape)}"
            )

        output = self.model.inference(model_input)

        logger.info(f"[MiVOLOAttr] raw output shape={tuple(output.shape)}")

        if self.model.meta.only_age:
            age_output = output
            gender = None
        else:
            age_output = output[:, 2]
            gender_output = output[:, :2].softmax(-1)
            _, gender_idx = gender_output.topk(1)
            gender = "male" if gender_idx[0].item() == 0 else "female"

        age = age_output[0].item()
        age = age * (self.model.meta.max_age - self.model.meta.min_age) + self.model.meta.avg_age
        age = round(age, 2)

        return age, gender

    def _to_gender(self, gender_value: Any) -> Gender:
        if gender_value is None:
            return Gender.unknown

        if isinstance(gender_value, (int, float)):
            if gender_value == 0:
                return Gender.male
            if gender_value == 1:
                return Gender.female

        gender_str = str(gender_value).strip().lower()

        if gender_str in ("male", "m", "0"):
            return Gender.male
        if gender_str in ("female", "f", "1"):
            return Gender.female

        return Gender.unknown

    def _to_age_group(self, age_value: Any) -> AgeGroup:
        if age_value is None:
            return AgeGroup.unknown

        try:
            age = float(age_value)
        except (TypeError, ValueError):
            return AgeGroup.unknown

        if 0 <= age <= 12:
            return AgeGroup.child
        if 13 <= age <= 19:
            return AgeGroup.teen
        if 20 <= age <= 29:
            return AgeGroup.young
        if 30 <= age <= 49:
            return AgeGroup.adult
        if age >= 50:
            return AgeGroup.senior

        return AgeGroup.unknown


# 이건 openvino 나이/성별 모델: https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/age-gender-recognition-retail-0013/FP32
# mivolo에서는 face+person(body) or face only 두 가지가 있는데 face+person이 gender에서 더 강하고 age도 비슷하거나 더 좋음
# model_imdb_cross_person_4.22_99.46.pth.tar

# MiVOLO repo 쓰는 이유; 전처리 함수, 모델 클래스, 체크포인트 로드 로직, age/gender 후처리를 src안에 직접 넣어야됨