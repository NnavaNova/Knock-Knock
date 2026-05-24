"""Fine-tune YOLO11 on the public CV training data.

Run this on the GCP Workbench, NOT inside the inference Docker image.
The output is a single file `cv/src/cv_finetuned.pt` that the inference
manager (cv_manager.py) auto-detects and prefers over the YOLO-World
fallback. On a closed 18-class taxonomy, a trained closed-vocab model
beats open-vocab YOLO-World by a wide margin — this is where the real
score jump comes from.

Usage on GCP Workbench:

    cd "$HOME/knock_knock_repo"
    pip install ultralytics==8.3.146      # if not already
    python cv/train.py                    # ~30-90 min on a single GPU
    git add cv/src/cv_finetuned.pt
    git commit -m "Add fine-tuned CV weights"
    git push origin main
    til build cv && til test cv && til submit cv

Override defaults with env vars:
    CV_TRAIN_DATA_DIR=/home/jupyter/novice/cv \
    CV_TRAIN_BASE=yolo11m.pt \
    CV_TRAIN_EPOCHS=100 \
    CV_TRAIN_IMGSZ=1280 \
    CV_TRAIN_BATCH=8 \
    python cv/train.py
"""

from __future__ import annotations

import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image
from ultralytics import YOLO


# Must match the 18-class layout the inference manager assumes. The order
# here IS the YOLO class index, which is also the challenge category_id.
CATEGORY_NAMES = [
    "cargo aircraft",
    "commercial aircraft",
    "drone",
    "fighter jet",
    "fighter plane",
    "helicopter",
    "light aircraft",
    "missile",
    "truck",
    "car",
    "tank",
    "bus",
    "van",
    "cargo ship",
    "yacht",
    "cruise ship",
    "warship",
    "sailboat",
]


def _norm_category_name(name: str) -> str:
    return " ".join(name.lower().replace("_", " ").split())


def _build_coco_id_to_yolo(ann: dict) -> dict[int, int]:
    """Map dataset category ids into the challenge's 0..17 class ids."""
    official_by_name = {
        _norm_category_name(name): idx for idx, name in enumerate(CATEGORY_NAMES)
    }
    aliases = {
        "cargo airplane": 0,
        "passenger airliner": 1,
        "airliner": 1,
        "fighter aircraft": 3,
        "military aircraft": 4,
        "small aircraft": 6,
        "small airplane": 6,
        "military tank": 10,
    }
    by_name = {**official_by_name, **aliases}
    observed_ids = sorted(
        {int(box["category_id"]) for box in ann.get("annotations", [])}
    )

    mapping: dict[int, int] = {}
    for entry in ann.get("categories", []):
        cat_id = int(entry.get("id"))
        name = _norm_category_name(entry.get("name") or "")
        if name in by_name:
            mapping[cat_id] = by_name[name]
        elif 0 <= cat_id < len(CATEGORY_NAMES):
            mapping[cat_id] = cat_id
        elif 1 <= cat_id <= len(CATEGORY_NAMES):
            mapping[cat_id] = cat_id - 1

    if not mapping:
        if observed_ids and min(observed_ids) >= 1 and max(observed_ids) <= len(CATEGORY_NAMES):
            mapping = {cat_id: cat_id - 1 for cat_id in observed_ids}
        else:
            mapping = {
                cat_id: cat_id
                for cat_id in observed_ids
                if 0 <= cat_id < len(CATEGORY_NAMES)
            }

    missing = sorted(set(observed_ids) - set(mapping))
    if missing:
        print(f"Warning: unmapped category ids will be skipped: {missing}")
    return mapping


def _resolve_data_dir() -> Path:
    """Find the public CV data directory on the Workbench."""
    explicit = os.environ.get("CV_TRAIN_DATA_DIR")
    if explicit:
        return Path(explicit)

    # Standard layouts on the Workbench. TEAM_TRACK is usually "novice"
    # or "advanced"; we probe both before giving up.
    candidates = []
    track = os.environ.get("TEAM_TRACK")
    if track:
        candidates.append(Path(f"/home/jupyter/{track}/cv"))
    candidates += [
        Path("/home/jupyter/novice/cv"),
        Path("/home/jupyter/advanced/cv"),
    ]
    for c in candidates:
        if (c / "annotations.json").exists() and (c / "images").is_dir():
            return c
    raise FileNotFoundError(
        "Could not locate the public CV dataset. Set CV_TRAIN_DATA_DIR to "
        "the directory containing annotations.json and images/."
    )


def _convert_coco_to_yolo(src_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    """Convert COCO-format annotations into the YOLO txt-per-image layout.

    YOLO expects:
        out_dir/images/train/*.jpg
        out_dir/labels/train/*.txt   # one row per object: cls cx cy w h (normalized)
        out_dir/images/val/*.jpg
        out_dir/labels/val/*.txt

    We use a deterministic class-stratified 90/10 train/val split so each
    category is represented in validation whenever the data permits it.
    """
    ann_path = src_dir / "annotations.json"
    images_src = src_dir / "images"
    with ann_path.open() as f:
        ann = json.load(f)

    coco_id_to_yolo = _build_coco_id_to_yolo(ann)
    print(f"Category mapping: {coco_id_to_yolo}")

    images_by_id: dict[int, dict] = {img["id"]: img for img in ann["images"]}
    boxes_by_image: dict[int, list[dict]] = defaultdict(list)
    for a in ann.get("annotations", []):
        boxes_by_image[a["image_id"]].append(a)

    image_classes: dict[int, set[int]] = {}
    class_to_images: dict[int, list[int]] = defaultdict(list)
    for img_id in sorted(images_by_id):
        classes = {
            coco_id_to_yolo[int(box["category_id"])]
            for box in boxes_by_image.get(img_id, [])
            if int(box["category_id"]) in coco_id_to_yolo
        }
        image_classes[img_id] = classes
        for cls_id in classes:
            class_to_images[cls_id].append(img_id)

    train_ids, val_ids = _stratified_split(
        sorted(images_by_id),
        image_classes,
        class_to_images,
        val_fraction=float(os.getenv("CV_TRAIN_VAL_FRACTION", "0.10")),
        seed=int(os.getenv("CV_TRAIN_SPLIT_SEED", "42")),
    )
    print(f"Train: {len(train_ids)} images   Val: {len(val_ids)} images")

    img_root = out_dir / "images"
    lbl_root = out_dir / "labels"
    for split in ("train", "val"):
        (img_root / split).mkdir(parents=True, exist_ok=True)
        (lbl_root / split).mkdir(parents=True, exist_ok=True)

    def _write_split(ids: list[int], split: str) -> None:
        for img_id in ids:
            meta = images_by_id[img_id]
            fname = meta["file_name"]
            src_img = images_src / fname
            if not src_img.exists():
                continue
            dst_img = img_root / split / fname
            if not dst_img.exists():
                try:
                    dst_img.symlink_to(src_img.resolve())
                except OSError:
                    shutil.copy2(src_img, dst_img)
            # Image dimensions for normalization.
            width = meta.get("width")
            height = meta.get("height")
            if not width or not height:
                with Image.open(src_img) as pil_img:
                    width, height = pil_img.size

            lines: list[str] = []
            for box in boxes_by_image.get(img_id, []):
                cat = coco_id_to_yolo.get(int(box["category_id"]))
                if cat is None:
                    continue
                x, y, w, h = box["bbox"]
                cx = (x + w / 2.0) / width
                cy = (y + h / 2.0) / height
                nw = w / width
                nh = h / height
                if nw <= 0 or nh <= 0:
                    continue
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                nw = min(max(nw, 0.0), 1.0)
                nh = min(max(nh, 0.0), 1.0)
                lines.append(f"{cat} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            label_path = lbl_root / split / (Path(fname).stem + ".txt")
            label_path.write_text("\n".join(lines))

    _write_split(train_ids, "train")
    _write_split(val_ids, "val")

    return img_root / "train", img_root / "val"


def _stratified_split(
    image_ids: list[int],
    image_classes: dict[int, set[int]],
    class_to_images: dict[int, list[int]],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    desired_val = max(1, round(len(image_ids) * val_fraction))
    class_totals = {cls_id: len(ids) for cls_id, ids in class_to_images.items()}
    train_counts = dict(class_totals)
    val_counts: dict[int, int] = defaultdict(int)
    val_set: set[int] = set()

    rng = random.Random(seed)

    def can_move(img_id: int) -> bool:
        classes = image_classes.get(img_id, set())
        if not classes:
            return True
        return all(train_counts.get(cls_id, 0) > 1 for cls_id in classes)

    def add_val(img_id: int) -> bool:
        if img_id in val_set or not can_move(img_id):
            return False
        val_set.add(img_id)
        for cls_id in image_classes.get(img_id, set()):
            train_counts[cls_id] -= 1
            val_counts[cls_id] += 1
        return True

    # Seed validation with rare classes first so every class has a measured AP.
    for cls_id in sorted(class_totals, key=lambda c: (class_totals[c], c)):
        if val_counts[cls_id] > 0:
            continue
        candidates = list(class_to_images[cls_id])
        rng.shuffle(candidates)
        candidates.sort(key=lambda img_id: (len(image_classes.get(img_id, set())), img_id))
        for img_id in candidates:
            if add_val(img_id):
                break

    candidates = list(image_ids)
    rng.shuffle(candidates)
    candidates.sort(key=lambda img_id: (img_id % 10, img_id))
    for img_id in candidates:
        if len(val_set) >= desired_val:
            break
        add_val(img_id)

    train_ids = [img_id for img_id in image_ids if img_id not in val_set]
    val_ids = [img_id for img_id in image_ids if img_id in val_set]
    if not train_ids:
        raise RuntimeError("Stratified split left no training images")
    if not val_ids:
        raise RuntimeError("Stratified split left no validation images")
    return train_ids, val_ids


def _write_dataset_yaml(out_dir: Path) -> Path:
    yaml_path = out_dir / "cv_dataset.yaml"
    config = {
        "path": str(out_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CATEGORY_NAMES)},
    }
    with yaml_path.open("w") as f:
        yaml.safe_dump(config, f)
    return yaml_path


def main() -> None:
    src_dir = _resolve_data_dir()
    print(f"Using training data from: {src_dir}")

    work_root = Path(
        os.environ.get("CV_TRAIN_WORK_DIR", str(Path.home() / "cv_train_work"))
    )
    work_root.mkdir(parents=True, exist_ok=True)
    print(f"Working directory: {work_root}")

    _convert_coco_to_yolo(src_dir, work_root)
    yaml_path = _write_dataset_yaml(work_root)
    print(f"Dataset config: {yaml_path}")

    # Defaults target Novice Workbench GPUs: yolo11m is strong enough for the
    # fixed 18-class taxonomy while leaving room for 1280px localization.
    base = os.environ.get("CV_TRAIN_BASE", "yolo11m.pt")
    epochs = int(os.environ.get("CV_TRAIN_EPOCHS", "20"))
    imgsz = int(os.environ.get("CV_TRAIN_IMGSZ", "1280"))
    batch = int(os.environ.get("CV_TRAIN_BATCH", "2"))
    workers = int(os.environ.get("CV_TRAIN_WORKERS", "4"))
    cache = os.environ.get("CV_TRAIN_CACHE", "disk")
    patience = int(os.environ.get("CV_TRAIN_PATIENCE", "6"))
    project = str(work_root / "runs")

    print(f"Fine-tuning {base} for {epochs} epochs @ imgsz={imgsz} batch={batch}")
    model = YOLO(base)
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name="cv_finetune",
        exist_ok=True,
        patience=patience,
        optimizer=os.getenv("CV_TRAIN_OPTIMIZER", "AdamW"),
        lr0=float(os.getenv("CV_TRAIN_LR0", "0.0015")),
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3.0,
        close_mosaic=10,
        mosaic=0.8,
        mixup=0.05,
        copy_paste=0.10,
        degrees=5.0,
        translate=0.08,
        scale=0.35,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.45,
        hsv_v=0.25,
        cache=cache,
        workers=workers,
        seed=int(os.getenv("CV_TRAIN_SEED", "42")),
        deterministic=False,
        amp=True,
        plots=True,
    )

    # Locate the best checkpoint, copy to the inference path inside the repo.
    best = Path(project) / "cv_finetune" / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Expected best.pt at {best} after training")

    out_path = Path(__file__).resolve().parent / "src" / "cv_finetuned.pt"
    shutil.copy2(best, out_path)
    print(f"\nFine-tuned weights written to: {out_path}")
    print("Next: commit cv/src/cv_finetuned.pt, push, then til build cv.")


if __name__ == "__main__":
    main()
