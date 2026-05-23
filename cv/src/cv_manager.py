"""Manages the CV model.

The detector resolves in this priority:
  1. A fine-tuned closed-vocab YOLO checkpoint at /workspace/cv_finetuned.pt
     (produced by cv/train.py on the public training data). This always
     beats open-vocab on a fixed taxonomy.
  2. YOLO-World v2 with single descriptive prompts per category. Used as the
     fallback when no fine-tuned model is shipped in the image.

Inference applies horizontal-flip test-time augmentation and merges the two
prediction sets with Weighted Box Fusion (WBF), which on COCO consistently
beats vanilla NMS by 1-3 mAP. The merger preserves localization quality
(the costly part of mAP@[0.5:0.95]) far better than simple max-over-overlaps.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any

from PIL import Image

LOGGER = logging.getLogger(__name__)

# 18 closed-set categories. For YOLO-World fallback, we use ONE specific
# noun phrase per category — alias lists confuse the text encoder by creating
# multiple competing embeddings for the same visual concept. These prompts
# were picked to be unambiguous and well-represented in LVIS-style data.
CATEGORY_PROMPTS = {
    0: "cargo airplane",
    1: "passenger airliner",
    2: "drone",
    3: "fighter jet",
    4: "military propeller airplane",
    5: "helicopter",
    6: "small private airplane",
    7: "missile",
    8: "truck",
    9: "car",
    10: "military tank",
    11: "bus",
    12: "van",
    13: "cargo ship",
    14: "yacht",
    15: "cruise ship",
    16: "warship",
    17: "sailboat",
}

# Index aligned with CATEGORY_PROMPTS — used to map YOLO model-class IDs
# back to the challenge's category_id space.
PROMPT_LIST = [CATEGORY_PROMPTS[i] for i in range(len(CATEGORY_PROMPTS))]


class CVManager:
    """Closed-vocab YOLO detector with TTA + WBF, falling back to YOLO-World."""

    def __init__(self):
        self.finetuned_path = Path(
            os.getenv("CV_FINETUNED_PATH", "/workspace/cv_finetuned.pt")
        )
        self.yolo_world_name = os.getenv("CV_MODEL_NAME", "yolov8x-worldv2.pt")
        self.image_size = int(os.getenv("CV_IMGSZ", "1280"))
        # Higher than the old 0.12 — the evaluator sets every detection's score
        # to 1.0, so false positives count against precision the same as hits.
        # We want to be more confident before submitting a box.
        self.confidence = float(os.getenv("CV_CONF", "0.30"))
        self.iou = float(os.getenv("CV_IOU", "0.55"))
        self.max_det = int(os.getenv("CV_MAX_DET", "60"))
        self.use_tta = os.getenv("CV_TTA", "1") != "0"
        self.wbf_iou = float(os.getenv("CV_WBF_IOU", "0.55"))
        self.wbf_skip_thr = float(os.getenv("CV_WBF_SKIP", "0.001"))
        self.model = None
        self.is_finetuned = False
        self.num_classes = len(CATEGORY_PROMPTS)
        self.load_error = ""
        self._tried_loading = False

    # ------------------------------------------------------------------ API

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        return self.cv_batch([image])[0]

    def warmup(self) -> None:
        self._ensure_model()

    def cv_batch(self, images: list[bytes]) -> list[list[dict[str, Any]]]:
        if not images:
            return []
        if not self._ensure_model():
            return [[] for _ in images]

        decoded = [self._decode_image(image) for image in images]
        predictions: list[list[dict[str, Any]]] = [[] for _ in images]

        valid_pairs = [
            (idx, img) for idx, img in enumerate(decoded) if img is not None
        ]
        if not valid_pairs:
            return predictions

        valid_indices = [idx for idx, _ in valid_pairs]
        valid_images = [img for _, img in valid_pairs]
        sizes = [img.size for img in valid_images]  # (width, height)

        # Original-orientation inference.
        orig_dets = self._predict(valid_images)

        # TTA: horizontal flip, run, then unflip the boxes back to original
        # coordinates. Merging the two views with WBF yields more stable
        # localization than either alone.
        if self.use_tta:
            flipped_images = [
                img.transpose(Image.FLIP_LEFT_RIGHT) for img in valid_images
            ]
            flipped_dets = self._predict(flipped_images)
            unflipped_dets = [
                self._unflip_boxes(dets, width)
                for dets, (width, _height) in zip(flipped_dets, sizes)
            ]
        else:
            unflipped_dets = [[] for _ in valid_images]

        for idx, original_idx in enumerate(valid_indices):
            width, height = sizes[idx]
            merged = self._wbf_merge(
                [orig_dets[idx], unflipped_dets[idx]], width, height
            )
            predictions[original_idx] = merged

        return predictions

    # ---------------------------------------------------------- model load

    def _ensure_model(self) -> bool:
        if self.model is not None:
            return True
        if self._tried_loading:
            return False
        self._tried_loading = True
        try:
            self._load_model()
            self.load_error = ""
        except Exception as exc:
            self.model = None
            self.load_error = str(exc)
            LOGGER.exception("CV model failed to load: %s", exc)
        return self.model is not None

    def _load_model(self) -> None:
        # 1) Prefer a fine-tuned closed-vocab YOLO if it's been baked into
        #    the image. This is the path that wins big on score.
        if self.finetuned_path.exists():
            try:
                from ultralytics import YOLO

                self.model = YOLO(str(self.finetuned_path))
                self.is_finetuned = True
                try:
                    self.model.fuse()
                except Exception:
                    pass
                LOGGER.info(
                    "Loaded fine-tuned CV model from %s", self.finetuned_path
                )
                return
            except Exception as exc:
                LOGGER.warning(
                    "Fine-tuned model load failed (%s); falling back to YOLO-World",
                    exc,
                )

        # 2) Fallback: YOLO-World with descriptive prompts.
        from ultralytics import YOLOWorld

        self.model = YOLOWorld(self.yolo_world_name)
        self.model.set_classes(PROMPT_LIST)
        self.is_finetuned = False
        try:
            self.model.fuse()
        except Exception:
            pass

    # --------------------------------------------------------- prediction

    def _predict(self, images: list[Image.Image]) -> list[list[dict[str, Any]]]:
        """Run the underlying model on a list of PIL images.

        Returns per-image detection dicts with x1/y1/x2/y2 + score + class_id
        in the challenge's category_id space.
        """
        try:
            results = self.model.predict(
                images,
                imgsz=self.image_size,
                conf=self.confidence,
                iou=self.iou,
                max_det=self.max_det,
                agnostic_nms=False,
                verbose=False,
            )
        except Exception as exc:
            LOGGER.exception("YOLO predict failed: %s", exc)
            return [[] for _ in images]

        per_image: list[list[dict[str, Any]]] = []
        for img, result in zip(images, results):
            width, height = img.size
            per_image.append(self._extract_detections(result, width, height))
        return per_image

    def _extract_detections(
        self, result: Any, image_width: int, image_height: int
    ) -> list[dict[str, Any]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        confs = boxes.conf.detach().cpu().numpy()

        dets: list[dict[str, Any]] = []
        for box, cls_id, conf in zip(xyxy, classes, confs):
            category_id = self._map_class_to_category(int(cls_id))
            if category_id is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            x1 = max(0.0, min(x1, image_width - 1.0))
            y1 = max(0.0, min(y1, image_height - 1.0))
            x2 = max(x1 + 1.0, min(x2, float(image_width)))
            y2 = max(y1 + 1.0, min(y2, float(image_height)))
            if x2 - x1 < 3.0 or y2 - y1 < 3.0:
                continue
            dets.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "score": float(conf),
                    "category_id": category_id,
                }
            )
        return dets

    def _map_class_to_category(self, cls_id: int) -> int | None:
        """Convert a YOLO model class index to the challenge category_id."""
        if self.is_finetuned:
            # The training script writes classes in 0..17 order matching the
            # challenge category space directly.
            if 0 <= cls_id < self.num_classes:
                return cls_id
            return None
        # YOLO-World: classes 0..17 align with PROMPT_LIST positions, which
        # are themselves in 0..17 category order.
        if 0 <= cls_id < self.num_classes:
            return cls_id
        return None

    # ------------------------------------------------------------ TTA glue

    @staticmethod
    def _unflip_boxes(
        dets: list[dict[str, Any]], image_width: int
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for det in dets:
            x1, x2 = det["x1"], det["x2"]
            new_x1 = image_width - x2
            new_x2 = image_width - x1
            out.append({**det, "x1": new_x1, "x2": new_x2})
        return out

    # ------------------------------------------------------------- WBF

    def _wbf_merge(
        self,
        det_lists: list[list[dict[str, Any]]],
        image_width: int,
        image_height: int,
    ) -> list[dict[str, Any]]:
        """Fuse multiple detection sets with Weighted Box Fusion.

        Each input is a per-detector detection list in absolute pixel coords;
        WBF works in normalized [0, 1] coords and returns one fused set with
        averaged box geometry weighted by per-source confidence. This produces
        sharper boxes than NMS and a noticeable mAP@[0.5:0.95] lift since
        higher IoU thresholds reward tight localization.

        Falls back gracefully to concatenation + per-class NMS if ensemble_boxes
        is unavailable.
        """
        all_empty = all(not dets for dets in det_lists)
        if all_empty:
            return []

        try:
            from ensemble_boxes import weighted_boxes_fusion
        except Exception:
            return self._fallback_merge(det_lists, image_width, image_height)

        boxes_per_model: list[list[list[float]]] = []
        scores_per_model: list[list[float]] = []
        labels_per_model: list[list[int]] = []
        w = float(image_width)
        h = float(image_height)
        for dets in det_lists:
            if not dets:
                boxes_per_model.append([])
                scores_per_model.append([])
                labels_per_model.append([])
                continue
            boxes_per_model.append(
                [
                    [
                        max(0.0, min(1.0, d["x1"] / w)),
                        max(0.0, min(1.0, d["y1"] / h)),
                        max(0.0, min(1.0, d["x2"] / w)),
                        max(0.0, min(1.0, d["y2"] / h)),
                    ]
                    for d in dets
                ]
            )
            scores_per_model.append([float(d["score"]) for d in dets])
            labels_per_model.append([int(d["category_id"]) for d in dets])

        try:
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                boxes_per_model,
                scores_per_model,
                labels_per_model,
                iou_thr=self.wbf_iou,
                skip_box_thr=self.wbf_skip_thr,
            )
        except Exception as exc:
            LOGGER.warning("WBF failed (%s); using fallback merge", exc)
            return self._fallback_merge(det_lists, image_width, image_height)

        out: list[dict[str, Any]] = []
        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            x1, y1, x2, y2 = [float(v) for v in box]
            left = x1 * w
            top = y1 * h
            right = x2 * w
            bottom = y2 * h
            bw = max(1.0, right - left)
            bh = max(1.0, bottom - top)
            if bw < 3.0 or bh < 3.0:
                continue
            out.append(
                {
                    "bbox": [
                        round(left, 2),
                        round(top, 2),
                        round(bw, 2),
                        round(bh, 2),
                    ],
                    "category_id": int(label),
                }
            )

        # Sort by descending score so the most-confident detections appear
        # first in the response. The evaluator ignores ordering, but this is
        # nicer for downstream debugging.
        return out

    def _fallback_merge(
        self,
        det_lists: list[list[dict[str, Any]]],
        image_width: int,
        image_height: int,
    ) -> list[dict[str, Any]]:
        """Per-class greedy NMS over the concatenation of all det lists."""
        all_dets: list[dict[str, Any]] = []
        for dets in det_lists:
            all_dets.extend(dets)
        all_dets.sort(key=lambda d: d["score"], reverse=True)

        kept: list[dict[str, Any]] = []
        for det in all_dets:
            collide = False
            for prev in kept:
                if prev["category_id"] != det["category_id"]:
                    continue
                if self._xyxy_iou(det, prev) > self.iou:
                    collide = True
                    break
            if not collide:
                kept.append(det)

        out: list[dict[str, Any]] = []
        for det in kept:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            if bw < 3.0 or bh < 3.0:
                continue
            out.append(
                {
                    "bbox": [
                        round(x1, 2),
                        round(y1, 2),
                        round(bw, 2),
                        round(bh, 2),
                    ],
                    "category_id": int(det["category_id"]),
                }
            )
        return out

    @staticmethod
    def _xyxy_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
        inter_x1 = max(a["x1"], b["x1"])
        inter_y1 = max(a["y1"], b["y1"])
        inter_x2 = min(a["x2"], b["x2"])
        inter_y2 = min(a["y2"], b["y2"])
        iw = max(0.0, inter_x2 - inter_x1)
        ih = max(0.0, inter_y2 - inter_y1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
        area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
        return inter / max(area_a + area_b - inter, 1e-6)

    # ---------------------------------------------------------- decoding

    @staticmethod
    def _decode_image(image: bytes) -> Image.Image | None:
        try:
            return Image.open(io.BytesIO(image)).convert("RGB")
        except Exception:
            return None
