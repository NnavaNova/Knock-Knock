"""Build a conservative ASR domain lexicon from public challenge text."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TERM_RE = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}|[A-Z]{2,}(?:-[A-Z0-9]+)*|"
    r"[A-Za-z]+(?:-[A-Za-z0-9]+)+)\b"
)


def _candidate_paths() -> list[Path]:
    paths = [ROOT / "nlp" / "nlp.jsonl", *sorted((ROOT / "nlp" / "documents").glob("*.txt"))]
    track = os.getenv("TEAM_TRACK", "novice")
    paths.extend(
        [
            Path(f"/home/jupyter/{track}/asr/asr.jsonl"),
            Path(f"/home/jupyter/{track}/nlp/nlp.jsonl"),
        ]
    )
    return [path for path in paths if path.exists()]


def _texts_from_path(path: Path) -> list[str]:
    if path.suffix == ".jsonl":
        texts = []
        with path.open(errors="ignore") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                texts.extend(str(row.get(key, "")) for key in ("question", "answer", "transcript", "text"))
        return texts
    return [path.read_text(errors="ignore")]


def _clean_term(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip(" .,:;()[]{}\"'"))


def main() -> None:
    counts: Counter[str] = Counter()
    for path in _candidate_paths():
        for text in _texts_from_path(path):
            for match in TERM_RE.finditer(text):
                term = _clean_term(match.group(0))
                if len(term) >= 5:
                    counts[term] += 1

    stop = {
        "Document Type",
        "Classification",
        "Distribution",
        "Executive Summary",
        "Background",
        "Analysis",
        "Conclusion",
    }
    terms = [
        term
        for term, count in counts.most_common()
        if term not in stop and (count >= 2 or any(ch.isupper() for ch in term[1:]))
    ][:800]
    phrases = [term for term in terms if " " in term][:300]
    single_terms = [term for term in terms if " " not in term][:500]

    out = ROOT / "asr" / "src" / "domain_lexicon.json"
    out.write_text(json.dumps({"terms": single_terms, "phrases": phrases}, indent=2))
    print(f"Wrote {out}: {len(single_terms)} terms, {len(phrases)} phrases")


if __name__ == "__main__":
    main()
