#!/usr/bin/env bash
set -euo pipefail

# Use this when long CV training is already producing good validation mAP and
# you want to stop early. It copies Ultralytics' current best checkpoint into
# the submission image, tunes thresholds, commits the artifact, builds, submits.

cd "${TIL_FOLDER:-$HOME/knock_knock_repo}"
export TIL_FOLDER="${TIL_FOLDER:-$PWD}"

: "${TEAM_TRACK:=novice}"
: "${CV_TRAIN_WORK_DIR:=$HOME/cv_train_work}"
: "${CV_TRAIN_DATA_DIR:=/home/jupyter/${TEAM_TRACK}/cv}"
: "${CV_TUNE_IMGSZ:=1280}"
: "${CV_TUNE_MAX_IMAGES:=300}"
: "${CV_SUBMIT_TAG:=cv-yolo11m-1280-early}"

BEST="${CV_TRAIN_WORK_DIR}/runs/cv_finetune/weights/best.pt"
if [[ ! -s "${BEST}" ]]; then
  echo "Missing ${BEST}."
  echo "Let training finish at least one validation pass, then run this again."
  exit 1
fi

mkdir -p cv/src
cp "${BEST}" cv/src/cv_finetuned.pt
echo "Copied current best checkpoint:"
ls -lh cv/src/cv_finetuned.pt

python -m pip install \
  ultralytics==8.3.146 \
  pycocotools \
  ensemble-boxes \
  pillow \
  pyyaml \
  tqdm

if [[ "${CV_SKIP_TUNE:-0}" != "1" ]]; then
  echo "Tuning thresholds on up to ${CV_TUNE_MAX_IMAGES} validation images"
  CV_TRAIN_DATA_DIR="${CV_TRAIN_DATA_DIR}" \
  CV_TUNE_IMGSZ="${CV_TUNE_IMGSZ}" \
  CV_TUNE_MAX_IMAGES="${CV_TUNE_MAX_IMAGES}" \
  python cv/tune_thresholds.py
else
  if [[ ! -s cv/src/cv_thresholds.json ]]; then
    python - <<'PY'
from pathlib import Path
import json
Path("cv/src/cv_thresholds.json").write_text(
    json.dumps({str(i): 0.30 for i in range(18)}, indent=2)
)
PY
  fi
fi

git add cv/src/cv_finetuned.pt cv/src/cv_thresholds.json
if ! git diff --cached --quiet; then
  git commit -m "Add early CV checkpoint"
  git push origin "$(git branch --show-current)"
fi

til build cv "${CV_SUBMIT_TAG}"
til submit cv "${CV_SUBMIT_TAG}"
