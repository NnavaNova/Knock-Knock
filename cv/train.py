"""Fine-tune YOLO11x on the public CV training data.

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
    CV_TRAIN_BASE=yolo11x.pt \
    CV_TRAIN_EPOCHS=60 \
    CV_TRAIN_IMGSZ=1280 \
    CV_TRAIN_BATCH=8 \
    python cv/train.py
"""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image
from ultralytics import YOLO


# Must match the 18-class layout the inference manager assumes. The order
# here IS the YOLO class index, which is also the challenge category_id.
CATEGORY_NAMES = [
    "cargo_airplane",
    "passenger_airliner",
    "drone",
    "fighter_jet",
    "military_propeller_airplane",
    "helicopter",
    "small_private_airplane",
    "missile",
    "truck",
    "car",
    "military_tank",
    "bus",
    "van",
    "cargo_ship",
    "yacht",
    "cruise_ship",
    "warship",
    "sailboat",
]


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

    We use a deterministic 90/10 train/val split by image id.
    """
    ann_path = src_dir / "annotations.json"
    images_src = src_dir / "images"
    with ann_path.open() as f:
        ann = json.load(f)

    # Build the COCO-id -> 0..17 category index. The dataset's category_id
    # ordering must match CATEGORY_NAMES, which by construction matches
    # the challenge category_id space, so this is straight identity in
    # practice — but we re-derive it defensively in case the annotations
    # JSON ever permutes categories.
    coco_cats = ann.get("categories", [])
    # If categories use the [0..17] ids, identity. Otherwise map by name.
    coco_id_to_yolo: dict[int, int] = {}
    if coco_cats:
        for entry in coco_cats:
            cat_id = entry.get("id")
            name = (entry.get("name") or "").lower().replace(" ", "_")
            if name in CATEGORY_NAMES:
                coco_id_to_yolo[int(cat_id)] = CATEGORY_NAMES.index(name)
            elif isinstance(cat_id, int) and 0 <= cat_id < len(CATEGORY_NAMES):
                # Annotations use raw 0..17 ids without names — trust the id.
                coco_id_to_yolo[int(cat_id)] = int(cat_id)
    else:
        # Annotations lack a categories array; assume raw 0..17 ids.
        coco_id_to_yolo = {i: i for i in range(len(CATEGORY_NAMES))}

    images_by_id: dict[int, dict] = {img["id"]: img for img in ann["images"]}
    boxes_by_image: dict[int, list[dict]] = defaultdict(list)
    for a in ann.get("annotations", []):
        boxes_by_image[a["image_id"]].append(a)

    # Deterministic split: image id modulo 10. Bucket 0 -> val (10%).
    train_ids: list[int] = []
    val_ids: list[int] = []
    for img_id in sorted(images_by_id):
        (val_ids if img_id % 10 == 0 else train_ids).append(img_id)
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

    # Defaults tuned for a T4 (~14.5 GB usable). yolo11x at batch=8 imgsz=1280
    # OOMs on T4 — use yolo11l (~25M params, only ~1 mAP behind 11x on COCO,
    # and the gap shrinks further with fine-tuning) at a slightly smaller
    # image size. If you have an A100/L4/H100 you can bump these up via env.
    base = os.environ.get("CV_TRAIN_BASE", "yolo11l.pt")
    epochs = int(os.environ.get("CV_TRAIN_EPOCHS", "60"))
    imgsz = int(os.environ.get("CV_TRAIN_IMGSZ", "1024"))
    batch = int(os.environ.get("CV_TRAIN_BATCH", "8"))
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
        patience=15,
        # Mosaic-off for the last few epochs sharpens small-object boxes —
        # important for our high-IoU metric.
        close_mosaic=10,
        # Modest augmentation; the corpus already has plenty of variation.
        mixup=0.10,
        copy_paste=0.10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # Optimizer-and-schedule defaults are good for YOLO11; only nudge LR.
        lr0=0.0025,
        cos_lr=True,
        plots=False,
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
