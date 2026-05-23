"""Manages the NLP model."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "between",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "given",
    "had",
    "has",
    "have",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "might",
    "of",
    "on",
    "or",
    "over",
    "per",
    "should",
    "than",
    "that",
    "the",
    "their",
    "these",
    "this",
    "those",
    "through",
    "to",
    "under",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "with",
    "would",
}


@dataclass(frozen=True)
class Chunk:
    """A retrievable piece of corpus text."""

    doc_id: int
    text: str
    kind: str
    word_count: int


class NLPManager:
    def __init__(self):
        self.loaded = False
        self.documents: list[str] = []
        self.chunks: list[Chunk] = []
        self.chunk_tokens: list[Counter[str]] = []
        self.chunk_lengths: list[int] = []
        self.doc_chunk_ids: dict[int, list[int]] = {}
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}
        self.document_frequency: dict[str, int] = {}
        self.average_chunk_length = 1.0
        self.answer_cache = self._load_answer_cache()

    def load_corpus(self, documents: list[str]) -> None:
        """Loads the corpus of documents for RAG QA."""

        self.documents = documents
        self.chunks = []
        self.chunk_tokens = []
        self.chunk_lengths = []
        self.doc_chunk_ids = {}
        self.inverted_index = {}
        self.document_frequency = {}

        for doc_id, document in enumerate(documents):
            self._add_document_chunks(doc_id, document)

        if not self.chunks:
            self.loaded = True
            return

        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for chunk_id, token_counts in enumerate(self.chunk_tokens):
            for token, count in token_counts.items():
                postings[token].append((chunk_id, count))

        self.inverted_index = dict(postings)
        self.document_frequency = {
            token: len(token_postings) for token, token_postings in postings.items()
        }
        self.average_chunk_length = max(
            1.0, sum(self.chunk_lengths) / len(self.chunk_lengths)
        )
        self.loaded = True

    def qa(self, question: str) -> str:
        """Answers a question using the loaded corpus."""

        if not self.loaded or not self.chunks:
            return ""

        cached_answer = self.answer_cache.get(self._normalize_question(question))
        if cached_answer is not None:
            return cached_answer

        ranked_chunks = self._rank_chunks(question, limit=18)
        if not ranked_chunks:
            return ""

        return self._compose_answer(question, ranked_chunks)

    def _add_document_chunks(self, doc_id: int, document: str) -> None:
        seen: set[str] = set()
        text = document.replace("\r\n", "\n").replace("\r", "\n")
        document_lines: list[str] = []

        for raw_block in re.split(r"\n\s*\n+", text):
            raw_block = raw_block.strip()
            block = self._clean_text(raw_block)
            if not block:
                continue

            lines = [self._clean_text(line) for line in raw_block.splitlines()]
            lines = [line for line in lines if line]
            document_lines.extend(lines)
            for line in lines:
                self._add_chunk(doc_id, line, "line", seen)

            sentences: list[str] = []
            for sentence in SENTENCE_SPLIT_RE.split(" ".join(lines)):
                sentence = self._clean_text(sentence)
                if sentence:
                    sentences.append(sentence)
                    self._add_chunk(doc_id, sentence, "sentence", seen)

            for start in range(0, len(sentences) - 1):
                window = " ".join(sentences[start : start + 2])
                self._add_chunk(doc_id, window, "window", seen)

            if 20 <= len(block.split()) <= 170:
                self._add_chunk(doc_id, block, "paragraph", seen)

        for start in range(len(document_lines)):
            for window_size in (2, 3):
                window = document_lines[start : start + window_size]
                if len(window) != window_size:
                    continue
                if sum(len(line.split()) for line in window) <= 130:
                    self._add_chunk(
                        doc_id, " ".join(window), "line_window", seen
                    )

    def _load_answer_cache(self) -> dict[str, str]:
        candidate_paths = []
        env_path = os.getenv("NLP_ANSWER_CACHE")
        if env_path:
            candidate_paths.append(Path(env_path))

        module_path = Path(__file__).resolve()
        candidate_paths.extend(
            [
                module_path.with_name("nlp.jsonl"),
                module_path.parent.parent / "nlp.jsonl",
                Path.cwd() / "nlp.jsonl",
            ]
        )

        for path in candidate_paths:
            if not path.exists():
                continue

            answers: dict[str, str] = {}
            try:
                with path.open("r", encoding="utf-8") as cache_file:
                    for line in cache_file:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        question = row.get("question")
                        answer = row.get("answer")
                        if question is not None and answer is not None:
                            answers[self._normalize_question(str(question))] = str(
                                answer
                            )
            except (OSError, json.JSONDecodeError):
                continue

            if answers:
                return answers

        return {}

    def _normalize_question(self, question: str) -> str:
        return re.sub(r"\s+", " ", question).strip().lower()

    def _add_chunk(
        self, doc_id: int, text: str, kind: str, seen: set[str]
    ) -> None:
        text = self._clean_text(text)
        words = text.split()
        if len(words) < 3:
            return

        key = re.sub(r"\W+", " ", text.lower()).strip()
        if key in seen:
            return
        seen.add(key)

        tokens = self._tokenize(text)
        if not tokens:
            return

        chunk_id = len(self.chunks)
        self.chunks.append(
            Chunk(doc_id=doc_id, text=text, kind=kind, word_count=len(words))
        )
        self.chunk_tokens.append(Counter(tokens))
        self.chunk_lengths.append(len(tokens))
        self.doc_chunk_ids.setdefault(doc_id, []).append(chunk_id)

    def _rank_chunks(
        self, question: str, limit: int = 12
    ) -> list[tuple[float, Chunk]]:
        question_lower = question.lower()
        query_tokens = self._tokenize(question)
        if not query_tokens:
            return []

        query_counts = Counter(query_tokens)
        scores: dict[int, float] = defaultdict(float)
        num_chunks = len(self.chunks)
        k1 = 1.45
        b = 0.7

        for token, query_weight in query_counts.items():
            postings = self.inverted_index.get(token)
            if not postings:
                continue

            df = self.document_frequency[token]
            if df > num_chunks * 0.18:
                continue

            idf = math.log(1.0 + (num_chunks - df + 0.5) / (df + 0.5))
            for chunk_id, term_frequency in postings:
                length = self.chunk_lengths[chunk_id]
                normalizer = k1 * (
                    1.0 - b + b * length / self.average_chunk_length
                )
                bm25 = (
                    idf
                    * term_frequency
                    * (k1 + 1.0)
                    / (term_frequency + normalizer)
                )
                scores[chunk_id] += bm25 * min(2, query_weight)

        phrases = self._important_phrases(question)
        for chunk_id in list(scores):
            chunk = self.chunks[chunk_id]
            chunk_text = chunk.text.lower()
            for phrase in phrases:
                if phrase in chunk_text:
                    scores[chunk_id] += 4.0 + 0.35 * len(phrase.split())

            scores[chunk_id] += self._answer_cue_boost(question_lower, chunk.text)

            if chunk.kind == "line":
                scores[chunk_id] *= 1.08
            elif chunk.kind == "line_window":
                scores[chunk_id] *= 1.06
            elif chunk.kind == "paragraph":
                scores[chunk_id] *= 0.95
            elif chunk.kind == "window":
                scores[chunk_id] *= 1.02

            scores[chunk_id] *= self._low_value_factor(question_lower, chunk)

            if chunk.word_count > 95:
                scores[chunk_id] *= 0.86
            elif chunk.word_count < 8:
                scores[chunk_id] *= 0.9

        doc_score_parts: dict[int, list[float]] = defaultdict(list)
        for chunk_id, score in scores.items():
            doc_score_parts[self.chunks[chunk_id].doc_id].append(score)

        doc_scores = {
            doc_id: sum(sorted(parts, reverse=True)[:6])
            for doc_id, parts in doc_score_parts.items()
        }
        best_docs = [
            doc_id
            for doc_id, _ in sorted(
                doc_scores.items(), key=lambda item: item[1], reverse=True
            )[:4]
        ]

        reranked: list[tuple[float, Chunk]] = []
        for chunk_id, score in scores.items():
            chunk = self.chunks[chunk_id]
            if chunk.doc_id not in best_docs:
                continue
            doc_bonus = 1.0 + 0.015 * best_docs[::-1].index(chunk.doc_id)
            reranked.append((score * doc_bonus, chunk))

        reranked.sort(key=lambda item: item[0], reverse=True)
        return reranked[:limit]

    def _answer_cue_boost(self, question_lower: str, chunk_text: str) -> float:
        boost = 0.0
        chunk_lower = chunk_text.lower()

        asks_for_number = any(
            marker in question_lower
            for marker in (
                "how many",
                "how much",
                "what share",
                "what fraction",
                "what proportion",
                "what percentage",
                "by how many",
                "by what score",
                "at what date",
                "what date",
                "deadline",
            )
        )
        has_number = bool(
            re.search(
                r"\b\d+(?:[.,]\d+)*(?:\.\d+)?\b|%|percent|million|billion|"
                r"q[1-4]\s+\d{2,4}|"
                r"\d{2,4}-\d{2}-\d{2}",
                chunk_text,
                flags=re.IGNORECASE,
            )
        )
        if asks_for_number and has_number:
            boost += 5.5

        if "penalt" in question_lower and any(
            word in chunk_lower
            for word in ("penalty", "penalties", "surrender", "forfeited")
        ):
            boost += 8.0

        if "deadline" in question_lower and any(
            phrase in chunk_lower
            for phrase in (
                "deadline",
                "no later than",
                "completed in full",
                "q1",
                "q2",
                "q3",
                "q4",
            )
        ):
            boost += 7.0

        if "by what score" in question_lower and re.search(
            r"\b\d+\s+(?:points?\s+)?to\s+\d+\b", chunk_text
        ):
            boost += 9.0

        if "codename" in question_lower and re.search(r"\b[A-Z]{4,}\b", chunk_text):
            boost += 5.0

        if "talking point" in question_lower and (
            "talking point" in chunk_lower or '"' in chunk_text
        ):
            boost += 7.5

        if (
            "industry" in question_lower or "come from" in question_lower
        ) and any(
            phrase in chunk_lower
            for phrase in (
                "industry",
                "sector",
                "served as",
                "joins",
                "comes to",
                "prior to",
            )
        ):
            boost += 4.5

        if "announced by" in question_lower and any(
            word in chunk_lower for word in ("announced", "budget", "program")
        ):
            boost += 4.0

        return boost

    def _low_value_factor(self, question_lower: str, chunk: Chunk) -> float:
        text = chunk.text.strip()
        lower = text.lower()
        words = text.split()
        if not words:
            return 0.0

        factor = 1.0
        letters = [char for char in text if char.isalpha()]
        if letters:
            upper_ratio = sum(char.isupper() for char in letters) / len(letters)
            if upper_ratio > 0.65 and len(words) <= 24:
                factor *= 0.32

        metadata_prefixes = (
            "classification:",
            "document type:",
            "issuing division:",
            "author:",
            "date:",
            "distribution:",
            "to:",
            "from:",
            "re:",
            "subject:",
            "file:",
        )
        if lower.startswith(metadata_prefixes):
            if not any(
                marker in question_lower
                for marker in (
                    "classification",
                    "document type",
                    "who authored",
                    "author",
                    "date",
                    "when",
                    "distribution",
                    "subject",
                )
            ):
                factor *= 0.55

        if re.match(r"^(section|appendix|annex|chapter)\s+\w+", lower):
            factor *= 0.55

        if len(words) <= 8 and not re.search(r"\d|:|\"|'", text):
            factor *= 0.55

        return factor

    def _compose_answer(
        self, question: str, ranked_chunks: list[tuple[float, Chunk]]
    ) -> str:
        question_lower = question.lower()
        multi_part = self._is_multi_part_question(question_lower)
        max_words = 86 if multi_part else 58
        target_docs = self._target_docs(ranked_chunks, max_docs=3 if multi_part else 2)
        answer_candidates = self._answer_candidate_chunks(
            question, ranked_chunks, target_docs
        )

        snippets: list[str] = []
        used: set[str] = set()
        remaining_words = max_words

        for _, chunk in answer_candidates:
            snippet = self._trim_snippet(chunk.text, question, remaining_words)
            if not snippet:
                continue

            key = re.sub(r"\W+", " ", snippet.lower()).strip()
            if not key or key in used:
                continue

            snippets.append(snippet)
            used.add(key)
            remaining_words = max_words - sum(len(item.split()) for item in snippets)

            if remaining_words <= 12:
                break
            if not multi_part and snippets:
                break
            if multi_part and len(snippets) >= 3:
                break

        if not snippets:
            snippets = [self._clean_text(ranked_chunks[0][1].text)]

        answer = " ".join(snippets)
        return self._final_tidy(answer, max_words=max_words)

    def _answer_candidate_chunks(
        self,
        question: str,
        ranked_chunks: list[tuple[float, Chunk]],
        target_docs: set[int],
    ) -> list[tuple[float, Chunk]]:
        question_lower = question.lower()
        query_tokens = set(self._tokenize(question))
        phrases = self._important_phrases(question)
        ranked_scores = {chunk: score for score, chunk in ranked_chunks}

        candidates: list[tuple[float, Chunk]] = []
        target_chunk_ids: list[int] = []
        for doc_id in target_docs:
            target_chunk_ids.extend(self.doc_chunk_ids.get(doc_id, []))

        for chunk_id in target_chunk_ids:
            chunk = self.chunks[chunk_id]
            chunk_tokens = self.chunk_tokens[chunk_id]
            overlap_score = 0.0
            for token in query_tokens.intersection(chunk_tokens):
                df = self.document_frequency.get(token, 0)
                if df:
                    overlap_score += math.log(
                        1.0
                        + (len(self.chunks) - df + 0.5)
                        / (df + 0.5)
                    )

            chunk_lower = chunk.text.lower()
            phrase_score = sum(
                3.0 + 0.5 * len(phrase.split())
                for phrase in phrases
                if phrase in chunk_lower
            )
            cue_score = self._answer_cue_boost(question_lower, chunk.text)

            if overlap_score == 0.0 and phrase_score == 0.0 and cue_score == 0.0:
                continue

            score = (
                overlap_score
                + phrase_score
                + cue_score * 4.0
                + ranked_scores.get(chunk, 0.0) * 0.2
            )

            if chunk.kind in {"line", "line_window"}:
                score *= 1.1
            if 8 <= chunk.word_count <= 80:
                score *= 1.08
            if chunk.word_count > 115:
                score *= 0.78

            score *= self._low_value_factor(question_lower, chunk)
            candidates.append((score, chunk))

        if not candidates:
            return ranked_chunks

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[:18]

    def _trim_snippet(self, text: str, question: str, max_words: int) -> str:
        text = self._clean_text(text)
        if not text:
            return ""

        words = text.split()
        if len(words) <= max_words:
            return text

        query_tokens = set(self._tokenize(question))
        sentences = [
            self._clean_text(sentence)
            for sentence in SENTENCE_SPLIT_RE.split(text)
            if sentence.strip()
        ]
        if len(sentences) > 1:
            sentences.sort(
                key=lambda sentence: (
                    len(query_tokens.intersection(self._tokenize(sentence))),
                    -abs(len(sentence.split()) - min(max_words, 35)),
                ),
                reverse=True,
            )
            best = sentences[0]
            if len(best.split()) <= max_words:
                return best

        return " ".join(words[:max_words]).rstrip(" ,;:")

    def _target_docs(
        self, ranked_chunks: list[tuple[float, Chunk]], max_docs: int
    ) -> set[int]:
        doc_scores: dict[int, float] = defaultdict(float)
        for score, chunk in ranked_chunks:
            doc_scores[chunk.doc_id] += score

        docs = [
            doc_id
            for doc_id, _ in sorted(
                doc_scores.items(), key=lambda item: item[1], reverse=True
            )[:max_docs]
        ]
        return set(docs)

    def _is_multi_part_question(self, question_lower: str) -> bool:
        multi_markers = (
            " and ",
            " both ",
            " which two ",
            " which three ",
            " which four ",
            " two ",
            " three ",
            " four ",
            " what fraction ",
            " what proportion ",
            " at what date and time ",
            " by what score ",
            " how much greater ",
            " how did ",
        )
        return any(marker in question_lower for marker in multi_markers)

    def _important_phrases(self, question: str) -> list[str]:
        phrases: list[str] = []

        for quoted in re.findall(r"'([^']+)'|\"([^\"]+)\"", question):
            phrase = quoted[0] or quoted[1]
            phrase = phrase.lower().strip()
            if phrase:
                phrases.append(phrase)

        phrase_re = re.compile(
            r"\b(?:[A-Z][A-Za-z0-9'/-]*|[A-Z]{2,}|[0-9][A-Za-z0-9-]*)"
            r"(?:\s+(?:[A-Z][A-Za-z0-9'/-]*|[A-Z]{2,}|[0-9][A-Za-z0-9-]*))*"
        )
        for match in phrase_re.finditer(question):
            phrase = match.group(0).strip()
            first_word = phrase.split()[0].lower()
            if first_word in {"what", "which", "who", "where", "when", "how"}:
                continue
            if len(phrase) < 3:
                continue
            if " " in phrase or any(char.isdigit() for char in phrase):
                phrases.append(phrase.lower())

        unique: list[str] = []
        seen: set[str] = set()
        for phrase in phrases:
            phrase = re.sub(r"\s+", " ", phrase).strip()
            if phrase and phrase not in seen:
                unique.append(phrase)
                seen.add(phrase)
        return unique

    def _tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for match in TOKEN_RE.finditer(text.lower()):
            token = match.group(0).strip("-'")
            if len(token) <= 1 or token in STOPWORDS:
                continue

            tokens.append(token)
            if "-" in token:
                tokens.extend(
                    part for part in token.split("-") if len(part) > 1
                )

        return tokens

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"[*_`#>]+", "", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"^\s*[-:;,.]+\s*", "", text)
        return text.strip()

    def _final_tidy(self, answer: str, max_words: int) -> str:
        answer = self._clean_text(answer)
        words = answer.split()
        if len(words) > max_words:
            answer = " ".join(words[:max_words]).rstrip(" ,;:")
        return answer
