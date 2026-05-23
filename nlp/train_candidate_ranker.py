"""Fit tiny heuristic answer-candidate weights from public NLP examples.

This intentionally does not ship or depend on the official evaluator model.
It uses public answers as weak labels to tune source preferences, then writes
`nlp/src/candidate_ranker.json` for runtime.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from nlp_manager import NLPManager  # noqa: E402


def _norm(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _match(candidate: str, answer: str) -> bool:
    cand = _norm(candidate)
    ref = _norm(answer)
    if not cand or not ref:
        return False
    return cand == ref or cand in ref or ref in cand


def _load_docs() -> list[dict[str, str]]:
    docs = []
    for path in sorted((ROOT / "nlp" / "documents").glob("DOC-*.txt")):
        docs.append({"id": path.stem, "document": path.read_text(errors="ignore")})
    return docs


def main() -> None:
    manager = NLPManager()
    manager.load_corpus(_load_docs())

    source_total: Counter[str] = Counter()
    source_hits: Counter[str] = Counter()
    lengths: defaultdict[str, list[int]] = defaultdict(list)

    with (ROOT / "nlp" / "nlp.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            question = row["question"]
            answer = row["answer"]
            ranked = manager._rank_chunks(question, limit=18)
            if not ranked:
                continue
            model_answer, model_score = manager._model_answer_with_score(question, ranked)
            direct_answer = manager._direct_answer(question, ranked)
            composed = manager._compose_answer(question, ranked)
            raw = [
                ("model", model_answer, 4.0 + max(-3.0, min(4.0, model_score))),
                ("direct", direct_answer, 8.0),
                ("snippet", composed, 2.0),
            ]
            for source, candidate, _score in raw:
                if not candidate:
                    continue
                source_total[source] += 1
                lengths[source].append(len(candidate.split()))
                if _match(candidate, answer):
                    source_hits[source] += 1

    weights: dict[str, float] = {"bias": 0.0, "length": -0.015, "short": 0.15}
    for source, total in source_total.items():
        rate = source_hits[source] / max(total, 1)
        weights[f"source:{source}"] = round((rate - 0.35) * 2.0, 4)
        avg_len = sum(lengths[source]) / max(len(lengths[source]), 1)
        print(source, "hit_rate", round(rate, 4), "avg_len", round(avg_len, 2))

    out_path = SRC / "candidate_ranker.json"
    out_path.write_text(json.dumps(weights, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
