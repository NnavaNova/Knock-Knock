#!/usr/bin/env bash
set -euo pipefail

# Final NLP run: build and submit the optimized runtime image directly from
# Workbench. No GitHub push is needed.

cd "${TIL_FOLDER:-$HOME/knock_knock_repo}"
export TIL_FOLDER="${TIL_FOLDER:-$PWD}"

: "${NLP_SUBMIT_TAG:=nlp-final-fast-v1}"

til build nlp "${NLP_SUBMIT_TAG}"
til submit nlp "${NLP_SUBMIT_TAG}"
