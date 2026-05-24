#!/usr/bin/env bash
set -euo pipefail

# Final CV run: no GitHub push from Workbench. This trains a stronger
# full-data YOLO11m checkpoint locally, uses less overfit thresholds, builds,
# and submits the Docker image directly from this folder.

cd "${TIL_FOLDER:-$HOME/knock_knock_repo}"
export TIL_FOLDER="${TIL_FOLDER:-$PWD}"

: "${TEAM_TRACK:=novice}"
: "${CV_TRAIN_DATA_DIR:=/home/jupyter/${TEAM_TRACK}/cv}"
: "${CV_TRAIN_BASE:=yolo11m.pt}"
: "${CV_TRAIN_EPOCHS:=20}"
: "${CV_TRAIN_IMGSZ:=1280}"
: "${CV_TRAIN_BATCH:=2}"
: "${CV_TRAIN_WORKERS:=4}"
: "${CV_TRAIN_CACHE:=disk}"
: "${CV_TRAIN_PATIENCE:=8}"
: "${CV_TRAIN_INCLUDE_VAL:=1}"
: "${CV_TRAIN_ERASING:=0.0}"
: "${CV_TUNE_MAX_THRESHOLD:=0.35}"
: "${CV_TUNE_MAX_IMAGES:=0}"
: "${CV_SUBMIT_TAG:=cv-final-yolo11m-e20}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -m pip install \
  ultralytics==8.3.146 \
  pycocotools \
  ensemble-boxes \
  pillow \
  pyyaml \
  tqdm

CV_TRAIN_DATA_DIR="${CV_TRAIN_DATA_DIR}" \
CV_TRAIN_BASE="${CV_TRAIN_BASE}" \
CV_TRAIN_EPOCHS="${CV_TRAIN_EPOCHS}" \
CV_TRAIN_IMGSZ="${CV_TRAIN_IMGSZ}" \
CV_TRAIN_BATCH="${CV_TRAIN_BATCH}" \
CV_TRAIN_WORKERS="${CV_TRAIN_WORKERS}" \
CV_TRAIN_CACHE="${CV_TRAIN_CACHE}" \
CV_TRAIN_PATIENCE="${CV_TRAIN_PATIENCE}" \
CV_TRAIN_INCLUDE_VAL="${CV_TRAIN_INCLUDE_VAL}" \
CV_TRAIN_ERASING="${CV_TRAIN_ERASING}" \
python cv/train.py

CV_TRAIN_DATA_DIR="${CV_TRAIN_DATA_DIR}" \
CV_TUNE_IMGSZ="${CV_TRAIN_IMGSZ}" \
CV_TUNE_MAX_THRESHOLD="${CV_TUNE_MAX_THRESHOLD}" \
CV_TUNE_MAX_IMAGES="${CV_TUNE_MAX_IMAGES}" \
python cv/tune_thresholds.py

ls -lh cv/src/cv_finetuned.pt cv/src/cv_thresholds.json

til build cv "${CV_SUBMIT_TAG}"
til submit cv "${CV_SUBMIT_TAG}"
