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
ENV CV_MODEL_NAME=yolov8x-worldv2.pt
ENV CV_IMGSZ=1280
ENV CV_CONF=0.12
ENV CV_IOU=0.55
ENV CV_MAX_DET=50

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from ultralytics import YOLOWorld; import os; YOLOWorld(os.environ['CV_MODEL_NAME'])"

# Copies your source files.
COPY src .

# Starts your model server.
CMD ["uvicorn", "cv_server:app", "--port", "5002", "--host", "0.0.0.0"]
