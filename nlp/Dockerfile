# Dockerfile for building the NLP image.


# The base image, an example deep learning VM.
# For a full list, see https://us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# For info, see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace
ENV QA_MODEL_NAME=deepset/deberta-v3-base-squad2
ENV EMB_MODEL_NAME=sentence-transformers/multi-qa-MiniLM-L6-cos-v1
ENV NLP_FAST_DIRECT=1
ENV NLP_MAX_MODEL_CONTEXTS=4
ENV NLP_MODEL_MAX_LENGTH=320
ENV NLP_MODEL_STRIDE=96
ENV NLP_MODEL_TOPK=8
ENV NLP_DENSE_CHUNKS=1

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from transformers import AutoModelForQuestionAnswering, AutoTokenizer; import os; name = os.environ['QA_MODEL_NAME']; AutoTokenizer.from_pretrained(name).save_pretrained('qa_model'); AutoModelForQuestionAnswering.from_pretrained(name).save_pretrained('qa_model')"
RUN python -c "from transformers import AutoModel, AutoTokenizer; import os; name = os.environ['EMB_MODEL_NAME']; AutoTokenizer.from_pretrained(name).save_pretrained('emb_model'); AutoModel.from_pretrained(name).save_pretrained('emb_model')"

# Copies your source files.
COPY src .

# Starts your model server.
CMD ["uvicorn", "nlp_server:app", "--port", "5004", "--host", "0.0.0.0"]
