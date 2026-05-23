"""Manages the ASR model."""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


LOGGER = logging.getLogger(__name__)


class ASRManager:
    """Whisper-based speech recognizer for the ASR challenge."""

    target_sample_rate = 16_000

    def __init__(self):
        self.model_name = os.getenv(
            "ASR_MODEL_NAME", "openai/whisper-large-v3-turbo"
        )
        self.language = os.getenv("ASR_LANGUAGE", "auto").strip().lower()
        self.batch_size = max(1, int(os.getenv("ASR_BATCH_SIZE", "4")))
        self.chunk_length_s = float(os.getenv("ASR_CHUNK_LENGTH_S", "30"))
        self.domain_terms, self.domain_phrases = self._load_domain_lexicon()
        self.pipe = None
        self._load_model()

    def asr(self, audio_bytes: bytes) -> str:
        """Performs ASR transcription on an audio file."""

        return self.asr_batch([audio_bytes])[0]

    def asr_batch(self, audio_items: list[bytes]) -> list[str]:
        """Transcribes a batch of WAV byte strings."""

        if not audio_items:
            return []
        if self.pipe is None:
            return ["" for _ in audio_items]

        inputs = [self._decode_audio(audio_bytes) for audio_bytes in audio_items]
        generate_kwargs = self._generate_kwargs()

        try:
            results = self._infer(inputs, generate_kwargs)
        except Exception as exc:
            LOGGER.exception("Batched ASR inference failed; retrying per clip: %s", exc)
            results = [self._infer_one(item, generate_kwargs) for item in inputs]

        if isinstance(results, dict):
            results = [results]

        return [self._result_text(result) for result in results]

    def _load_model(self) -> None:
        model_paths = [
            Path("/workspace/asr_model"),
            Path(__file__).resolve().parent / "asr_model",
            Path.cwd() / "asr_model",
            Path(self.model_name),
        ]
        model_path = next((path for path in model_paths if path.exists()), None)
        model_id = str(model_path) if model_path is not None else self.model_name

        try:
            import torch
            from transformers import (
                AutoModelForSpeechSeq2Seq,
                AutoProcessor,
                pipeline,
            )
        except Exception as exc:
            raise RuntimeError("ASR dependencies are not installed") from exc

        try:
            device_index = 0 if torch.cuda.is_available() else -1
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            model.config.forced_decoder_ids = None
            model.generation_config.forced_decoder_ids = None
            model.to(device)
            model.eval()

            processor = AutoProcessor.from_pretrained(model_id)
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device_index,
                chunk_length_s=self.chunk_length_s,
                stride_length_s=(4, 2),
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load ASR model from {model_id}") from exc

    def _generate_kwargs(self) -> dict:
        kwargs = {
            "task": "transcribe",
            "num_beams": int(os.getenv("ASR_NUM_BEAMS", "5")),
            "temperature": 0.0,
            "condition_on_prev_tokens": False,
            "max_new_tokens": int(os.getenv("ASR_MAX_NEW_TOKENS", "160")),
        }
        if self.language and self.language != "auto":
            kwargs["language"] = self.language
        return kwargs

    def _infer(self, inputs: list[dict], generate_kwargs: dict):
        return self.pipe(
            inputs,
            batch_size=min(self.batch_size, len(inputs)),
            generate_kwargs=generate_kwargs,
        )

    def _infer_one(self, item: dict, generate_kwargs: dict) -> dict:
        fallback_kwargs = [
            generate_kwargs,
            {
                key: generate_kwargs[key]
                for key in ("task", "language")
                if key in generate_kwargs
            },
            {},
        ]
        for kwargs in fallback_kwargs:
            try:
                return self.pipe(item, generate_kwargs=kwargs)
            except Exception as exc:
                LOGGER.warning("ASR retry failed with kwargs %s: %s", kwargs, exc)
        return {"text": ""}

    def _result_text(self, result: object) -> str:
        if isinstance(result, dict):
            return self._clean_transcript(str(result.get("text", "")))
        if isinstance(result, str):
            return self._clean_transcript(result)
        return ""

    def _decode_audio(self, audio_bytes: bytes) -> dict:
        try:
            audio, sample_rate = sf.read(
                io.BytesIO(audio_bytes), dtype="float32", always_2d=False
            )
        except Exception:
            return {
                "array": np.zeros(self.target_sample_rate, dtype=np.float32),
                "sampling_rate": self.target_sample_rate,
            }

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        elif audio.ndim > 2:
            audio = audio.reshape(-1)

        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if sample_rate != self.target_sample_rate and audio.size:
            gcd = math.gcd(int(sample_rate), self.target_sample_rate)
            audio = resample_poly(
                audio,
                self.target_sample_rate // gcd,
                int(sample_rate) // gcd,
            ).astype(np.float32)

        audio = self._trim_silence(audio)
        audio = self._normalize_audio(audio)
        return {"array": audio, "sampling_rate": self.target_sample_rate}

    def _trim_silence(self, audio: np.ndarray) -> np.ndarray:
        if audio.size < self.target_sample_rate // 2:
            return audio

        sr = self.target_sample_rate
        frame = int(0.02 * sr)
        hop = int(0.01 * sr)
        if audio.size <= frame:
            return audio

        rms = []
        for start in range(0, audio.size - frame + 1, hop):
            chunk = audio[start:start + frame]
            rms.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))

        rms_arr = np.asarray(rms, dtype=np.float32)
        if rms_arr.size == 0:
            return audio

        threshold = max(float(np.percentile(rms_arr, 80)) * 0.05, 1e-4)
        active = np.where(rms_arr > threshold)[0]
        if active.size == 0:
            return audio

        pad = int(0.20 * sr)
        start = max(0, int(active[0]) * hop - pad)
        end = min(audio.size, int(active[-1]) * hop + frame + pad)
        return audio[start:end]

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return np.zeros(self.target_sample_rate, dtype=np.float32)

        audio = audio.astype(np.float32, copy=False)
        audio = audio - float(np.mean(audio))

        peak = float(np.max(np.abs(audio)))
        if peak > 1e-5:
            audio = audio / peak * 0.95
        return np.ascontiguousarray(audio, dtype=np.float32)

    def _clean_transcript(self, transcript: str) -> str:
        transcript = re.sub(r"<\|[^>]+?\|>", " ", transcript)
        transcript = re.sub(r"\[[^\]]{0,40}\]", " ", transcript)
        transcript = re.sub(r"\([^)]{0,40}\)", " ", transcript)
        transcript = re.sub(r"\s+", " ", transcript)
        return self._domain_correct(transcript.strip())

    def _load_domain_lexicon(self) -> tuple[list[str], list[str]]:
        path = Path(__file__).resolve().parent / "domain_lexicon.json"
        if not path.exists():
            return [], []
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            LOGGER.warning("Could not load ASR lexicon: %s", exc)
            return [], []
        terms = sorted(
            {
                str(term).strip()
                for term in data.get("terms", [])
                if len(str(term).strip()) >= 5
            },
            key=len,
            reverse=True,
        )
        phrases = sorted(
            {
                str(phrase).strip()
                for phrase in data.get("phrases", [])
                if len(str(phrase).strip().split()) >= 2
            },
            key=len,
            reverse=True,
        )
        return terms, phrases

    def _domain_correct(self, transcript: str) -> str:
        if not transcript or not (self.domain_terms or self.domain_phrases):
            return transcript
        try:
            from rapidfuzz import fuzz, process
        except Exception:
            return transcript

        corrected = transcript
        phrase_terms = {
            self._score_normalize(phrase): phrase
            for phrase in self.domain_phrases
            if self._score_normalize(phrase)
        }
        if phrase_terms:
            phrase_candidates = list(phrase_terms)
            words = re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", corrected)
            for ngram_size in (4, 3, 2):
                if len(words) < ngram_size:
                    continue
                for start in range(0, len(words) - ngram_size + 1):
                    window = words[start:start + ngram_size]
                    norm = self._score_normalize(" ".join(window))
                    if len(norm) < 8:
                        continue
                    best = process.extractOne(norm, phrase_candidates, scorer=fuzz.ratio)
                    if not best:
                        continue
                    candidate, score, _idx = best
                    if score < 95 or candidate[:1] != norm[:1]:
                        continue
                    pattern = re.compile(
                        r"\b"
                        + r"[\s-]+".join(re.escape(part) for part in window)
                        + r"\b",
                        flags=re.IGNORECASE,
                    )
                    corrected = pattern.sub(phrase_terms[candidate], corrected, count=1)

        token_terms = {
            self._score_normalize(term): term
            for term in self.domain_terms
            if " " not in self._score_normalize(term)
        }
        if not token_terms:
            return corrected

        candidates = list(token_terms)

        def replace_token(match: re.Match[str]) -> str:
            token = match.group(0)
            norm = self._score_normalize(token)
            if len(norm) < 5:
                return token
            best = process.extractOne(norm, candidates, scorer=fuzz.ratio)
            if not best:
                return token
            candidate, score, _idx = best
            if score >= 92 and candidate[:1] == norm[:1]:
                return token_terms[candidate]
            return token

        return re.sub(r"\b[A-Za-z][A-Za-z'-]{4,}\b", replace_token, corrected)

    @staticmethod
    def _score_normalize(text: str) -> str:
        text = text.lower().replace("-", " ")
        text = re.sub(r"[^a-z0-9 ]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()
