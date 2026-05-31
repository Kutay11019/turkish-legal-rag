from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def simple_tokenize(text: str) -> List[str]:
    text = str(text).lower()
    text = re.sub(r"[^a-zçğıöşü0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    min_value = values.min()
    max_value = values.max()

    if max_value - min_value < 1e-9:
        return np.zeros_like(values)

    return (values - min_value) / (max_value - min_value)


class HybridRetriever:
    def __init__(
        self,
        chunks: List[Dict[str, str]],
        embedding_model_name: str,
        hybrid_alpha: float = 0.65,
    ) -> None:
        if not chunks:
            raise ValueError("Chunks cannot be empty.")

        self.chunks = chunks
        self.embedding_model_name = embedding_model_name
        self.hybrid_alpha = hybrid_alpha

        self.texts = [chunk["text"] for chunk in chunks]
        self.tokenized_texts = [simple_tokenize(text) for text in self.texts]

        self.bm25 = BM25Okapi(self.tokenized_texts)

        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embeddings = self.embedding_model.encode(
            self.texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> List[Dict[str, object]]:
        query_tokens = simple_tokenize(query)

        bm25_scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=float)

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        dense_scores = cosine_similarity(query_embedding, self.embeddings)[0]

        bm25_norm = minmax_normalize(bm25_scores)
        dense_norm = minmax_normalize(dense_scores)

        hybrid_scores = (
            self.hybrid_alpha * dense_norm
            + (1.0 - self.hybrid_alpha) * bm25_norm
        )

        candidate_k = min(candidate_k, len(self.chunks))
        top_k = min(top_k, candidate_k)

        candidate_indices = np.argsort(hybrid_scores)[::-1][:candidate_k]
        final_indices = candidate_indices[:top_k]

        results = []

        for rank, idx in enumerate(final_indices, start=1):
            chunk = self.chunks[int(idx)].copy()

            chunk.update(
                {
                    "rank": rank,
                    "dense_score": float(dense_scores[int(idx)]),
                    "bm25_score": float(bm25_scores[int(idx)]),
                    "hybrid_score": float(hybrid_scores[int(idx)]),
                }
            )

            results.append(chunk)

        return results