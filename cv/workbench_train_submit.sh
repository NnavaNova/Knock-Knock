#!/usr/bin/env bash
set -euo pipefail

# One-command CV recovery path for the GCP Workbench.
# This trains a closed-vocabulary YOLO checkpoint, tunes the confidence
# thresholds, commits the resulting weight/threshold files, builds, and submits.

cd "${TIL_FOLDER:-$HOME/knock_knock_repo}"
export TIL_FOLDER="${TIL_FOLDER:-$PWD}"

: "${TEAM_TRACK:=novice}"
: "${CV_TRAIN_DATA_DIR:=/home/jupyter/${TEAM_TRACK}/cv}"
: "${CV_TRAIN_BASE:=yolo11m.pt}"
: "${CV_TRAIN_EPOCHS:=100}"
: "${CV_TRAIN_IMGSZ:=1280}"
: "${CV_TRAIN_BATCH:=8}"
: "${CV_SUBMIT_TAG:=cv-yolo11m-1280-e100}"

python -m pip install -U pip
python -m pip install \
  ultralytics==8.3.146 \
  pycocotools \
  ensemble-boxes \
  pillow \
  pyyaml \
  tqdm

echo "Training CV model from ${CV_TRAIN_DATA_DIR}"
CV_TRAIN_DATA_DIR="${CV_TRAIN_DATA_DIR}" \
CV_TRAIN_BASE="${CV_TRAIN_BASE}" \
CV_TRAIN_EPOCHS="${CV_TRAIN_EPOCHS}" \
CV_TRAIN_IMGSZ="${CV_TRAIN_IMGSZ}" \
CV_TRAIN_BATCH="${CV_TRAIN_BATCH}" \
python cv/train.py

echo "Tuning per-class confidence thresholds"
CV_TRAIN_DATA_DIR="${CV_TRAIN_DATA_DIR}" \
CV_TUNE_IMGSZ="${CV_TRAIN_IMGSZ}" \
python cv/tune_thresholds.py

python - <<'PY'
from pathlib import Path
weights = Path("cv/src/cv_finetuned.pt")
thresholds = Path("cv/src/cv_thresholds.json")
if not weights.exists():
    raise SystemExit("missing cv/src/cv_finetuned.pt after training")
if weights.stat().st_size > 95 * 1024 * 1024:
    raise SystemExit(
        f"{weights} is {weights.stat().st_size / 1024 / 1024:.1f} MB; "
        "use yolo11m.pt or GitHub may reject the push"
    )
if not thresholds.exists():
    raise SystemExit("missing cv/src/cv_thresholds.json after tuning")
print(f"weights: {weights.stat().st_size / 1024 / 1024:.1f} MB")
PY

git add cv/src/cv_finetuned.pt cv/src/cv_thresholds.json
if ! git diff --cached --quiet; then
  git commit -m "Add tuned CV checkpoint"
  git push origin "$(git branch --show-current)"
fi

til build cv "${CV_SUBMIT_TAG}"

if [[ "${CV_SKIP_TEST:-1}" != "1" ]]; then
  til test cv "${CV_SUBMIT_TAG}"
fi

til submit cv "${CV_SUBMIT_TAG}"
