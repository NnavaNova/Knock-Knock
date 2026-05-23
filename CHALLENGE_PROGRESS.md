# Knock-Knock Challenge Progress

Last updated: 2026-05-23

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
`answer`.

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
til build nlp
til test nlp
til submit nlp
```

## ASR

Important commits:

```text
d3e0976 Fix ASR inference fallback
004d776 Improve ASR multilingual inference
```

The ASR route contract is unchanged: `/asr` takes base64 WAV inputs and returns
`{"predictions": ["..."]}`. The latest ASR model uses Whisper large-v3 turbo,
auto language detection, chunked inference, and safer fallback handling.

Run on Workbench:

```bash
git pull origin main
til build asr
til test asr
til submit asr
```

## CV

The official CV contract is `/cv` on port `5002`, input JPEG base64, output:

```json
{"predictions": [[{"bbox": [l, t, w, h], "category_id": 0}]]}
```

The submitted bbox format must be zero-indexed LTWH. Empty scenes return an
empty list for that image.

Current implementation uses YOLO-World with the exact 18 challenge classes:

```text
cargo aircraft, commercial aircraft, drone, fighter jet, fighter plane,
helicopter, light aircraft, missile, truck, car, tank, bus, van, cargo ship,
yacht, cruise ship, warship, sailboat
```

Run on Workbench:

```bash
git pull origin main
til build cv
til test cv
til submit cv
```
