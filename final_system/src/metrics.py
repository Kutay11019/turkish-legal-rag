from __future__ import annotations

import difflib
import re
from typing import Dict, List


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zçğıöşü0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    ref_counts = {}

    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0

    for token in pred_tokens:
        if ref_counts.get(token, 0) > 0:
            overlap += 1
            ref_counts[token] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def text_similarity(prediction: str, reference: str) -> float:
    return difflib.SequenceMatcher(
        None,
        normalize_text(prediction),
        normalize_text(reference),
    ).ratio()


def has_source_citation(answer: str) -> int:
    answer = str(answer).lower()
    return int("kaynak:" in answer or "citation:" in answer)


def summarize_results(rows: List[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {
            "sample_count": 0,
            "mean_token_f1": 0.0,
            "mean_text_similarity": 0.0,
            "source_citation_rate": 0.0,
        }

    token_f1_values = [float(row["token_f1"]) for row in rows]
    similarity_values = [float(row["text_similarity"]) for row in rows]
    citation_values = [float(row["has_source_citation"]) for row in rows]

    return {
        "sample_count": len(rows),
        "mean_token_f1": sum(token_f1_values) / len(token_f1_values),
        "mean_text_similarity": sum(similarity_values) / len(similarity_values),
        "source_citation_rate": sum(citation_values) / len(citation_values),
    }