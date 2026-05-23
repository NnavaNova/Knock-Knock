# Knock-Knock Challenge Progress

Last updated: 2026-05-24

## Shared Workbench Setup

Use the path without spaces for all `til` commands:

```bash
cd "$HOME/knock_knock_repo"
export TIL_FOLDER="$HOME/knock_knock_repo"
```

The GCP service account is authenticated, but Docker may still need Artifact
Registry auth before `til submit` can push:

```bash
gcloud auth configure-docker asia-southeast1-docker.pkg.dev --quiet
```

If that still fails:

```bash
gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://asia-southeast1-docker.pkg.dev
```

## NLP

Important commits:

```text
12d77bc Update NLP contract for document IDs
7b4848f Improve NLP short answer extraction
```

The official NLP input/output contract changed. The server now accepts corpus
documents as dicts with IDs and returns prediction dicts with `documents` and
`answer`. The latest server returns `loaded` after corpus processing, blends
dense chunk retrieval with BM25, and selects among short answer candidates.

Latest local direct-manager proxy after extraction improvements:

```text
retrieval_overlap: 853/883 = 0.9660
exact_norm:        115/883 = 0.1302
contains_norm:     281/883 = 0.3182
avg_similarity:    0.2914
```

Run on Workbench:

```bash
git pull origin main
python nlp/train_candidate_ranker.py
til build nlp chunkdense-ranker-v1
til test nlp chunkdense-ranker-v1
til submit nlp chunkdense-ranker-v1
```

## ASR

Important commits:

```text
d3e0976 Fix ASR inference fallback
004d776 Improve ASR multilingual inference
```

The ASR route contract is unchanged: `/asr` takes base64 WAV inputs and returns
`{"predictions": ["..."]}`. The latest ASR model uses Whisper large-v3 turbo,
English decoding for Novice, silence trimming, chunked inference, safer
fallback handling, and a generated domain lexicon.

Run on Workbench:

```bash
git pull origin main
python scripts/build_asr_lexicon.py
til build asr english-lexicon-v1
til test asr english-lexicon-v1
til submit asr english-lexicon-v1
```

## CV

The official CV contract is `/cv` on port `5002`, input JPEG base64, output:

```json
{"predictions": [[{"bbox": [l, t, w, h], "category_id": 0}]]}
```

The submitted bbox format must be zero-indexed LTWH. Empty scenes return an
empty list for that image.

Current implementation prefers a trained closed-vocabulary YOLO checkpoint at
`cv/src/cv_finetuned.pt`. If the checkpoint is absent, it falls back to
YOLO-World with the exact 18 challenge classes:

```text
cargo aircraft, commercial aircraft, drone, fighter jet, fighter plane,
helicopter, light aircraft, missile, truck, car, tank, bus, van, cargo ship,
yacht, cruise ship, warship, sailboat
```

Run on Workbench:

```bash
git pull origin main
pip install ultralytics==8.3.146 ensemble-boxes pycocotools
CV_TRAIN_DATA_DIR=/home/jupyter/novice/cv CV_TRAIN_BASE=yolo11m.pt CV_TRAIN_EPOCHS=100 CV_TRAIN_IMGSZ=1280 CV_TRAIN_BATCH=8 python cv/train.py
python cv/tune_thresholds.py
git add cv/src/cv_finetuned.pt cv/src/cv_thresholds.json
git commit -m "Add tuned CV weights and thresholds"
git push origin main
til build cv ft-yolo11m-1280-e100
til test cv ft-yolo11m-1280-e100
til submit cv ft-yolo11m-1280-e100
```

## AE

Important commits:

```text
8aa3bae AE: replace action=0 stub with priority-cascade Bomberman agent
1b48f82 AE: add stateful map planner
```

Latest official AE evaluation:

```text
Submission time: 2026-05-23
Score: 0.399
Speed: 0.555
```

The current AE implementation is a stateful Bomberman planner. It tracks known
walls, destructible walls, bases, bombs, visible enemies, collectibles,
respawns, and visited cells, then uses one BFS map per step for safety, target
routing, bombing opportunities, and frontier exploration.

Run on Workbench:

```bash
git pull origin main
python ae/learn_fixed_map.py
git add ae/src/fixed_map.py
git commit -m "Add learned AE novice fixed map"
git push origin main
til build ae fixed-map-v2
til test ae fixed-map-v2
til submit ae fixed-map-v2
```
