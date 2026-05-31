from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF
from docx import Document


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    doc = fitz.open(path)
    pages = []

    for page in doc:
        pages.append(page.get_text("text"))

    doc.close()

    return "\n".join(pages)


def read_docx(path: Path) -> str:
    document = Document(path)
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def load_single_document(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return read_txt(path)

    if suffix == ".pdf":
        return read_pdf(path)

    if suffix == ".docx":
        return read_docx(path)

    raise ValueError(f"Unsupported file type: {path}")


def load_documents(documents_dir: str | Path) -> List[Dict[str, str]]:
    documents_dir = Path(documents_dir)

    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

    documents = []

    for path in sorted(documents_dir.rglob("*")):
        if not path.is_file():
            continue

        if path.name.startswith("."):
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = load_single_document(path).strip()

        if not text:
            continue

        doc_id = path.stem.replace(" ", "_")

        documents.append(
            {
                "doc_id": doc_id,
                "file_name": path.name,
                "path": str(path),
                "text": text,
            }
        )

    if not documents:
        raise ValueError(
            f"No supported documents found in {documents_dir}. "
            f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    return documents