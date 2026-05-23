"""Manages the CV model."""

from __future__ import annotations

import io
import logging
import os
from typing import Any

from PIL import Image

LOGGER = logging.getLogger(__name__)

TARGET_CLASSES = {
    0: ("cargo aircraft", "cargo plane", "transport aircraft"),
    1: ("commercial aircraft", "passenger aircraft", "airliner", "airplane"),
    2: ("drone", "uav", "unmanned aerial vehicle", "quadcopter"),
    3: ("fighter jet", "jet fighter", "military jet"),
    4: ("fighter plane", "military aircraft", "warplane"),
    5: ("helicopter", "chopper"),
    6: ("light aircraft", "small airplane", "private aircraft"),
    7: ("missile", "rocket projectile"),
    8: ("truck", "lorry"),
    9: ("car", "automobile"),
    10: ("tank", "armored tank", "military tank"),
    11: ("bus", "coach bus"),
    12: ("van", "minivan"),
    13: ("cargo ship", "container ship", "freighter"),
    14: ("yacht", "motor yacht"),
    15: ("cruise ship", "passenger ship"),
    16: ("warship", "naval ship", "military ship"),
    17: ("sailboat", "sailing boat"),
}

MODEL_CLASSES = [
    class_name
    for class_aliases in TARGET_CLASSES.values()
    for class_name in class_aliases
]
MODEL_CLASS_TO_CATEGORY = [
    category_id
    for category_id, class_aliases in TARGET_CLASSES.items()
    for _ in class_aliases
]


class CVManager:
    """YOLO-World detector configured for the challenge target classes."""

    def __init__(self):
        self.model_name = os.getenv("CV_MODEL_NAME", "yolov8x-worldv2.pt")
        self.image_size = int(os.getenv("CV_IMGSZ", "1280"))
        self.confidence = float(os.getenv("CV_CONF", "0.12"))
        self.iou = float(os.getenv("CV_IOU", "0.55"))
        self.max_det = int(os.getenv("CV_MAX_DET", "50"))
        self.model = None
        self.load_error = ""
        self._tried_loading = False

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection on an image."""

        return self.cv_batch([image])[0]

    def warmup(self) -> None:
        """Best-effort model load for container startup."""

        self._ensure_model()

    def cv_batch(self, images: list[bytes]) -> list[list[dict[str, Any]]]:
        """Performs object detection on a batch of images."""

        if not images:
            return []
        if not self._ensure_model():
            return [[] for _ in images]

        decoded_images = [self._decode_image(image) for image in images]
        valid_pairs = [
            (index, image)
            for index, image in enumerate(decoded_images)
            if image is not None
        ]
        predictions: list[list[dict[str, Any]]] = [[] for _ in images]
        if not valid_pairs:
            return predictions

        valid_indices = [index for index, _ in valid_pairs]
        valid_images = [image for _, image in valid_pairs]
        try:
            results = self.model.predict(
                valid_images,
                imgsz=self.image_size,
                conf=self.confidence,
                iou=self.iou,
                max_det=self.max_det,
                agnostic_nms=True,
                verbose=False,
            )
        except Exception as exc:
            LOGGER.exception("CV prediction failed: %s", exc)
            return predictions

        if not results:
            return predictions

        for index, result in zip(valid_indices, results):
            image = decoded_images[index]
            if image is None:
                continue
            width, height = image.size
            detections = self._format_detections(result, width, height)
            predictions[index] = self._dedupe_detections(detections)
        return predictions

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLOWorld
        except Exception as exc:
            raise RuntimeError("CV dependencies are not installed") from exc

        try:
            self.model = YOLOWorld(self.model_name)
            self.model.set_classes(MODEL_CLASSES)
            try:
                self.model.fuse()
            except Exception:
                pass
        except Exception as exc:
            raise RuntimeError(f"Failed to load CV model {self.model_name}") from exc

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

    def _decode_image(self, image: bytes) -> Image.Image | None:
        try:
            return Image.open(io.BytesIO(image)).convert("RGB")
        except Exception:
            return None

    def _format_detections(
        self, result: Any, image_width: int, image_height: int
    ) -> list[dict[str, Any]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        confs = boxes.conf.detach().cpu().numpy()

        detections: list[dict[str, Any]] = []
        for box, model_class_id, confidence in zip(xyxy, classes, confs):
            if not 0 <= int(model_class_id) < len(MODEL_CLASS_TO_CATEGORY):
                continue
            category_id = MODEL_CLASS_TO_CATEGORY[int(model_class_id)]

            left, top, right, bottom = [float(value) for value in box]
            left = max(0.0, min(left, image_width - 1.0))
            top = max(0.0, min(top, image_height - 1.0))
            right = max(left + 1.0, min(right, float(image_width)))
            bottom = max(top + 1.0, min(bottom, float(image_height)))
            box_width = right - left
            box_height = bottom - top

            if box_width < 3.0 or box_height < 3.0:
                continue

            detections.append(
                {
                    "bbox": [
                        round(left, 2),
                        round(top, 2),
                        round(box_width, 2),
                        round(box_height, 2),
                    ],
                    "category_id": int(category_id),
                    "_confidence": float(confidence),
                }
            )

        detections.sort(key=lambda item: item["_confidence"], reverse=True)
        return detections

    def _dedupe_detections(
        self, detections: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for detection in detections:
            if any(
                detection["category_id"] == previous["category_id"]
                and self._ltwh_iou(detection["bbox"], previous["bbox"]) > 0.78
                for previous in kept
            ):
                continue
            kept.append(detection)

        for detection in kept:
            detection.pop("_confidence", None)
        return kept

    def _ltwh_iou(self, first: list[float], second: list[float]) -> float:
        first_left, first_top, first_width, first_height = first
        second_left, second_top, second_width, second_height = second

        first_right = first_left + first_width
        first_bottom = first_top + first_height
        second_right = second_left + second_width
        second_bottom = second_top + second_height

        inter_left = max(first_left, second_left)
        inter_top = max(first_top, second_top)
        inter_right = min(first_right, second_right)
        inter_bottom = min(first_bottom, second_bottom)
        inter_width = max(0.0, inter_right - inter_left)
        inter_height = max(0.0, inter_bottom - inter_top)
        intersection = inter_width * inter_height
        if intersection <= 0:
            return 0.0

        first_area = first_width * first_height
        second_area = second_width * second_height
        return intersection / max(first_area + second_area - intersection, 1e-6)
