# src/models/mivolo_attr.py
# infer(frame, tracks) -> Dict[track_id, PersonAttr]

from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import sys

import numpy as np
import torch
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
            "weights",
            "weights/age_gender/model_imdb_cross_person_4.22_99.46.pth.tar",
        )
        self.repo_root = cfg.get("repo_root", os.getenv("MIVOLO_REPO_ROOT", "MiVOLO"))
        self.min_face_size = int(cfg.get("min_face_size", 20))
        self.min_person_size = int(cfg.get("min_person_size", 40))
        self.use_persons = bool(cfg.get("use_persons", True))

        self.model = self._load_model()

        # track_id별 캐시: 나이/성별은 트랙 생존 기간 중 거의 안 바뀌므로
        # 한 번 추론에 성공하면 재사용해서 매 프레임 재추론을 피한다.
        self._attr_cache: Dict[int, PersonAttr] = {}

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
        각 track의 attr 필드를 채워서 반환.
        캐시에 없는 track만 face/person crop을 모아 한 번의 배치 추론으로 처리하고,
        캐시에 있으면 그 값을 그대로 재사용합니다.
        """
        valid_tracks: List[Track] = []
        face_imgs: List[np.ndarray] = []
        person_imgs: List[np.ndarray] = []

        for track in tracks:
            cached = self._attr_cache.get(track.track_id)
            if cached is not None:
                track.attr = cached
                continue

            face_img = self._crop_face(frame, track)
            if face_img is None:
                continue

            person_img = None
            if self.use_persons:
                person_img = self._crop_person(frame, track)
                if person_img is None:
                    continue

            valid_tracks.append(track)
            face_imgs.append(face_img)
            if self.use_persons:
                person_imgs.append(person_img)

        if valid_tracks:
            try:
                preds = self._predict_batch(face_imgs, person_imgs if self.use_persons else None)
            except Exception:
                # logger.exception("[MiVOLOAttr] batch prediction failed")
                preds = None

            if preds is not None:
                for track, pred in zip(valid_tracks, preds):
                    if pred is None:
                        continue
                    age, gender_value = pred
                    attr = PersonAttr(
                        gender=self._to_gender(gender_value),
                        age_group=self._to_age_group(age),
                    )
                    track.attr = attr
                    self._attr_cache[track.track_id] = attr

        # 트래커가 더 이상 반환하지 않는(=사라진) track_id는 캐시에서 정리
        active_ids = {t.track_id for t in tracks}
        for tid in list(self._attr_cache.keys()):
            if tid not in active_ids:
                del self._attr_cache[tid]

        return tracks

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

    def _predict_batch(
        self,
        face_imgs: List[np.ndarray],
        person_imgs: Optional[List[np.ndarray]] = None,
    ):
        """
        여러 face crop(+ person crop)을 한 번의 배치로 MiVOLO 모델에 넣어 age/gender 추론
        """
        from mivolo.data.misc import prepare_classification_images

        faces_input = prepare_classification_images(
            face_imgs,
            self.model.input_size,
            self.model.data_config["mean"],
            self.model.data_config["std"],
            device=self.model.device,
        )

        if faces_input is None:
            raise ValueError("prepare_classification_images returned None for faces")

        if self.use_persons:
            if person_imgs is None:
                raise ValueError("person_imgs is required when use_persons=True")

            persons_input = prepare_classification_images(
                person_imgs,
                self.model.input_size,
                self.model.data_config["mean"],
                self.model.data_config["std"],
                device=self.model.device,
            )

            if persons_input is None:
                raise ValueError("prepare_classification_images returned None for persons")

            # face + person => [B, 6, H, W]
            model_input = torch.cat([faces_input, persons_input], dim=1)
        else:
            model_input = faces_input

        output = self.model.inference(model_input)

        if self.model.meta.only_age:
            age_output = output
            genders = [None] * age_output.shape[0]
        else:
            age_output = output[:, 2]
            gender_output = output[:, :2].softmax(-1)
            _, gender_idx = gender_output.topk(1)
            genders = ["male" if idx.item() == 0 else "female" for idx in gender_idx]

        results = []
        for i in range(age_output.shape[0]):
            age = age_output[i].item()
            age = age * (self.model.meta.max_age - self.model.meta.min_age) + self.model.meta.avg_age
            results.append((round(age, 2), genders[i]))

        return results

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

        if 0 <= age <= 9:
            return AgeGroup.age_0_9
        if 10 <= age <= 19:
            return AgeGroup.age_10_19
        if 20 <= age <= 29:
            return AgeGroup.age_20_29
        if 30 <= age <= 39:
            return AgeGroup.age_30_39
        if 40 <= age <= 49:
            return AgeGroup.age_40_49
        if 50 <= age <= 59:
            return AgeGroup.age_50_59
        if age >= 60:
            return AgeGroup.age_60_plus

        return AgeGroup.unknown


# 이건 openvino 나이/성별 모델: https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/age-gender-recognition-retail-0013/FP32
# mivolo에서는 face+person(body) or face only 두 가지가 있는데 face+person이 gender에서 더 강하고 age도 비슷하거나 더 좋음
# model_imdb_cross_person_4.22_99.46.pth.tar

# MiVOLO repo 쓰는 이유; 전처리 함수, 모델 클래스, 체크포인트 로드 로직, age/gender 후처리를 src안에 직접 넣어야됨