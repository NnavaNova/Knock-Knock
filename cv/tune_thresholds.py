"""Tune per-class CV confidence thresholds on the local validation split.

Run after `python cv/train.py` has created `cv/src/cv_finetuned.pt`.
The evaluator treats every submitted detection as score=1.0, so thresholding is
the main precision/recall control we have at runtime.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from ultralytics import YOLO

from train import (
    CATEGORY_NAMES,
    _build_coco_id_to_yolo,
    _resolve_data_dir,
    _stratified_split,
)


if os.getenv("CV_TUNE_THRESHOLDS"):
    THRESHOLDS = [
        float(value.strip())
        for value in os.environ["CV_TUNE_THRESHOLDS"].split(",")
        if value.strip()
    ]
else:
    THRESHOLDS = [round(x / 100, 2) for x in range(5, 80, 5)]


def _load_validation_data() -> tuple[dict, list[int], dict[int, Path], dict[int, int]]:
    src_dir = _resolve_data_dir()
    with (src_dir / "annotations.json").open() as f:
        ann = json.load(f)

    coco_id_to_yolo = _build_coco_id_to_yolo(ann)
    boxes_by_image: dict[int, list[dict]] = {}
    image_classes: dict[int, set[int]] = {}
    class_to_images: dict[int, list[int]] = {}
    for image in ann["images"]:
        image_classes[int(image["id"])] = set()
    for box in ann.get("annotations", []):
        img_id = int(box["image_id"])
        cls_id = coco_id_to_yolo.get(int(box["category_id"]))
        if cls_id is None:
            continue
        boxes_by_image.setdefault(img_id, []).append(box)
        image_classes.setdefault(img_id, set()).add(cls_id)
        class_to_images.setdefault(cls_id, []).append(img_id)

    _train_ids, val_ids = _stratified_split(
        sorted(int(image["id"]) for image in ann["images"]),
        image_classes,
        class_to_images,
        val_fraction=float(os.getenv("CV_TUNE_VAL_FRACTION", "0.10")),
        seed=int(os.getenv("CV_TRAIN_SPLIT_SEED", "42")),
    )
    max_images = int(os.getenv("CV_TUNE_MAX_IMAGES", "0"))
    if max_images > 0 and len(val_ids) > max_images:
        val_ids = sorted(val_ids)[:max_images]

    image_paths = {
        int(image["id"]): src_dir / "images" / image["file_name"]
        for image in ann["images"]
    }
    return ann, val_ids, image_paths, coco_id_to_yolo


def _val_ground_truth(
    ann: dict, val_ids: list[int], coco_id_to_yolo: dict[int, int], out_path: Path
) -> COCO:
    val_set = set(val_ids)
    converted_annotations = []
    for box in ann.get("annotations", []):
        if int(box["image_id"]) not in val_set:
            continue
        mapped = coco_id_to_yolo.get(int(box["category_id"]))
        if mapped is None:
            continue
        converted = dict(box)
        converted["category_id"] = mapped
        converted_annotations.append(converted)
    gt = {
        "images": [img for img in ann["images"] if int(img["id"]) in val_set],
        "annotations": converted_annotations,
        "categories": [
            {"id": idx, "name": name}
            for idx, name in enumerate(CATEGORY_NAMES)
        ],
    }
    out_path.write_text(json.dumps(gt))
    return COCO(str(out_path))


def _predict_raw(model_path: Path, val_ids: list[int], image_paths: dict[int, Path]):
    model = YOLO(str(model_path))
    imgsz = int(os.getenv("CV_TUNE_IMGSZ", os.getenv("CV_IMGSZ", "1280")))
    detections: list[dict] = []
    for img_id in tqdm(val_ids, desc="predict"):
        path = image_paths[img_id]
        result = model.predict(
            str(path),
            imgsz=imgsz,
            conf=0.01,
            iou=float(os.getenv("CV_IOU", "0.55")),
            max_det=int(os.getenv("CV_MAX_DET", "120")),
            verbose=False,
        )[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        confs = boxes.conf.detach().cpu().numpy()
        for box, cls_id, conf in zip(xyxy, classes, confs):
            if not 0 <= int(cls_id) < 18:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            detections.append(
                {
                    "image_id": img_id,
                    "category_id": int(cls_id),
                    "bbox": [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)],
                    "score": float(conf),
                }
            )
    return detections


def _score(coco_gt: COCO, detections: list[dict]) -> float:
    if not detections:
        return 0.0
    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return float(evaluator.stats[0])


def main() -> None:
    model_path = Path(os.getenv("CV_TUNE_MODEL", "cv/src/cv_finetuned.pt"))
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    ann, val_ids, image_paths, coco_id_to_yolo = _load_validation_data()
    raw_detections = _predict_raw(model_path, val_ids, image_paths)
    thresholds = {cls_id: 0.30 for cls_id in range(18)}

    with tempfile.TemporaryDirectory() as tmp:
        coco_gt = _val_ground_truth(
            ann, val_ids, coco_id_to_yolo, Path(tmp) / "val_gt.json"
        )
        best_score = _score(
            coco_gt,
            [
                {**det, "score": 1.0}
                for det in raw_detections
                if det["score"] >= thresholds[det["category_id"]]
            ],
        )
        print(f"Initial mAP: {best_score:.4f}")

        for cls_id in range(18):
            cls_best_threshold = thresholds[cls_id]
            cls_best_score = best_score
            for threshold in THRESHOLDS:
                trial = dict(thresholds)
                trial[cls_id] = threshold
                filtered = [
                    {**det, "score": 1.0}
                    for det in raw_detections
                    if det["score"] >= trial[det["category_id"]]
                ]
                score = _score(coco_gt, filtered)
                if score > cls_best_score:
                    cls_best_score = score
                    cls_best_threshold = threshold
            thresholds[cls_id] = cls_best_threshold
            best_score = cls_best_score
            print(f"class {cls_id}: threshold={cls_best_threshold:.2f} mAP={best_score:.4f}")

    out_path = Path("cv/src/cv_thresholds.json")
    out_path.write_text(json.dumps({str(k): v for k, v in thresholds.items()}, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
