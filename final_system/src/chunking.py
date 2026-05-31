from __future__ import annotations

import re
from typing import Dict, List


def normalize_whitespace(text: str) -> str:
    text = str(text)
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[str]:
    text = normalize_whitespace(text)

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - chunk_overlap)

    return chunks


def build_chunks(
    documents: List[Dict[str, str]],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> List[Dict[str, str]]:
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(
            doc["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for idx, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}_chunk_{idx:04d}",
                    "doc_id": doc["doc_id"],
                    "file_name": doc["file_name"],
                    "text": chunk,
                }
            )

    return all_chunks