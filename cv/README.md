# CV

Your CV challenge is to detect and classify objects in an image.

This Readme provides a brief overview of the interface format; see the Wiki for the full [challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications).

## Input

The input is sent via a POST request to the `/cv` route on port 5002. It is a JSON document structured as such:

```JSON
{
  "instances": [
    {
      "key": 0,
      "b64": "BASE64_ENCODED_IMAGE"
    },
    ...
  ]
}
```

The `b64` key of each object in the `instances` list contains the base64-encoded bytes of the input image in JPEG format. The length of the `instances` list is variable.

## Output

Your route handler function must return a `dict` with this structure:

```Python
{
    "predictions": [
        [
            {
                "bbox": [x, y, w, h],
                "category_id": category_id
            },
            ...
        ],
        ...
    ]
}
```

where `x`, `y`, `w`, `h`, and `category_id` are defined as above.

If your model detects no objects in a scene, your handler should output an empty list for that scene.

The $k$-th element of `predictions` must be the prediction corresponding to the $k$-th element of `instances` for all $1 \le k \le n$, where n is the number of input instances. The length of `predictions` must equal that of `instances`.

## Detector strategy

The container picks the detector at startup in this priority:

1. `/workspace/cv_finetuned.pt` — a closed-vocab YOLO checkpoint fine-tuned on the public training data. This always beats open-vocab on a fixed 18-class taxonomy and is where the score jump comes from.
2. YOLO-World v2 (`yolov8x-worldv2.pt`) with single descriptive prompts per category. Used when no fine-tuned model is shipped.

Inference applies horizontal-flip test-time augmentation and merges the two views with [Weighted Box Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) (`ensemble-boxes` library). WBF preserves precise localization better than NMS, which matters because the evaluator uses COCO mAP@[0.5:0.95] — half the score is at IoU ≥ 0.75.

Note: the evaluator sets every detection's confidence to 1.0, so the model's `conf` threshold is the only thing keeping false positives out. Default is `CV_CONF=0.30`; lower it if recall is the bottleneck.

## Training the closed-vocab detector

Fast path on the GCP Workbench:

```bash
cd "$HOME/knock_knock_repo"
export TIL_FOLDER="$HOME/knock_knock_repo"
git pull origin main
bash cv/workbench_train_submit.sh
```

Manual path on the GCP Workbench (not in Docker):

```bash
cd "$HOME/knock_knock_repo"
pip install ultralytics==8.3.146 pycocotools ensemble-boxes
CV_TRAIN_DATA_DIR=/home/jupyter/novice/cv \
CV_TRAIN_BASE=yolo11m.pt \
CV_TRAIN_EPOCHS=20 \
CV_TRAIN_IMGSZ=1280 \
CV_TRAIN_BATCH=2 \
python cv/train.py            # ~30-90 min on a single GPU
python cv/tune_thresholds.py
git add cv/src/cv_finetuned.pt cv/src/cv_thresholds.json
git commit -m "Add tuned CV checkpoint"
git push origin main
til build cv cv-yolo11m-1280-e20
til submit cv cv-yolo11m-1280-e20
```

Knobs (env vars to `python cv/train.py`):
- `CV_TRAIN_DATA_DIR` — defaults to `/home/jupyter/{TEAM_TRACK}/cv`
- `CV_TRAIN_BASE` — default `yolo11m.pt`; use this first because it stays below GitHub's normal 100 MB file limit
- `CV_TRAIN_EPOCHS` (20), `CV_TRAIN_IMGSZ` (1280), `CV_TRAIN_BATCH` (2 on a T4)

The script does a deterministic class-stratified train/val split so every class is represented in validation when the data permits it.

If a long run is already going and validation mAP is good, stop after a few
epochs and submit the current best checkpoint:

```bash
cd "$HOME/knock_knock_repo"
export TIL_FOLDER="$HOME/knock_knock_repo"
bash cv/submit_current_best.sh
```
