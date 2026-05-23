# Dockerfile for building the CV image.


# The base image, an example deep learning VM.
# For a full list, see https://us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# For info, see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace
ENV YOLO_CONFIG_DIR=/tmp/ultralytics
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV CV_MODEL_NAME=yolov8x-worldv2.pt
ENV CV_IMGSZ=1280
# The evaluator scores every detection at 1.0, so false positives count as
# heavily as true positives against precision. Stay conservative.
ENV CV_CONF=0.30
ENV CV_IOU=0.55
ENV CV_MAX_DET=60
ENV CV_TTA=auto
ENV CV_WBF_IOU=0.55
ENV CV_WBF_SKIP=0.001
# Where the manager looks for a fine-tuned closed-vocab YOLO checkpoint.
# Drop a cv_finetuned.pt next to the source files (and add to repo) to
# upgrade the detector from open-vocab to trained closed-vocab.
ENV CV_FINETUNED_PATH=/workspace/cv_finetuned.pt

# Installs system libraries required by OpenCV, which Ultralytics imports.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the YOLO-World fallback weights so first-call latency stays low.
# Use the same descriptive single-prompt-per-category list that cv_manager.py
# applies at runtime; this is what gets cached into the model file.
RUN python -c "from ultralytics import YOLOWorld; import os; model = YOLOWorld(os.environ['CV_MODEL_NAME']); model.set_classes(['cargo airplane','passenger airliner','drone','fighter jet','military propeller airplane','helicopter','small private airplane','missile','truck','car','military tank','bus','van','cargo ship','yacht','cruise ship','warship','sailboat'])"

# Copies your source files. If cv_finetuned.pt is included in src/, the
# manager will prefer it over the YOLO-World fallback.
COPY src .

# Starts your model server.
CMD ["uvicorn", "cv_server:app", "--port", "5002", "--host", "0.0.0.0"]
