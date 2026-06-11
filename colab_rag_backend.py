from __future__ import annotations

import hashlib
import io
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from docx import Document
except Exception:
    Document = None

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# =========================
# Report-aligned best pipeline constants
# =========================
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DOCS_DIR = ROOT_DIR / "final_system" / "data" / "custom_documents"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TURKISH_BGE_RERANKER_NAME = "seroe/bge-reranker-v2-m3-turkish-triplet"
BASE_MISTRAL_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

CANDIDATE_K = 10
FINAL_CONTEXT_K = 3
ALPHA = 0.5
HYBRID_WEIGHT = 0.7
RERANK_WEIGHT = 0.3
ARTICLE_BONUS_WEIGHT = 0.15
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MIN_FILTERED_ROWS = 1
SUPPORTED_UPLOAD_TYPES = {".txt", ".pdf", ".docx"}

app = FastAPI(title="Turkish Legal RAG GPU Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory model/index cache for the Colab session
_CACHE: Dict[str, object] = {}


def normalize_whitespace(text: str) -> str:
    text = str(text).replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def simple_turkish_tokenize(text: str) -> List[str]:
    text = str(text).lower()
    text = re.sub(r"[^a-zçğıöşü0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) == 0:
        return scores
    if float(scores.max()) == float(scores.min()):
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


def read_txt_from_bytes(data: bytes) -> str:
    for enc in ["utf-8", "utf-8-sig", "cp1254", "latin-1"]:
        try:
            return data.decode(enc, errors="ignore")
        except Exception:
            pass
    return data.decode("utf-8", errors="ignore")


def read_pdf_from_bytes(data: bytes) -> str:
    if fitz is None:
        raise ImportError("PDF okumak için PyMuPDF gerekli: pip install pymupdf")
    pdf = fitz.open(stream=data, filetype="pdf")
    pages = [page.get_text("text") for page in pdf]
    pdf.close()
    return "\n".join(pages)


def read_docx_from_bytes(data: bytes) -> str:
    if Document is None:
        raise ImportError("DOCX okumak için python-docx gerekli: pip install python-docx")
    doc = Document(io.BytesIO(data))
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


def read_file_bytes(file_name: str, data: bytes) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".txt":
        return read_txt_from_bytes(data)
    if suffix == ".pdf":
        return read_pdf_from_bytes(data)
    if suffix == ".docx":
        return read_docx_from_bytes(data)
    raise ValueError(f"Unsupported file type: {file_name}")


def read_file_path(path: Path) -> str:
    return read_file_bytes(path.name, path.read_bytes())


LAW_SOURCE_RULES = [
    ("Türkiye Cumhuriyeti Anayasası", ["anayasa", "constitution", "1982 anayasası"]),
    ("Türk Ceza Kanunu", ["tck", "türk ceza kanunu", "ceza kanunu"]),
    ("Ceza Muhakemesi Kanunu", ["cmk", "ceza muhakemesi kanunu"]),
    ("Türk Medeni Kanunu", ["tmk", "medeni kanun", "türk medeni kanunu"]),
    ("Türk Borçlar Kanunu", ["tbk", "borçlar kanunu", "türk borçlar kanunu"]),
    ("Hukuk Muhakemeleri Kanunu", ["hmk", "hukuk muhakemeleri kanunu"]),
    ("İdari Yargılama Usulü Kanunu", ["iyuk", "idari yargılama usulü"]),
    ("Kişisel Verilerin Korunması Kanunu", ["kvkk", "kişisel verilerin korunması"]),
    ("Bilgi Edinme Hakkı Kanunu", ["bilgi edinme hakkı", "bilgi edinme"]),
]


def infer_source_name(file_name: str, text: str) -> str:
    haystack = f"{file_name}\n{text[:3000]}".lower()
    for source_name, keywords in LAW_SOURCE_RULES:
        if any(keyword in haystack for keyword in keywords):
            return source_name
    return Path(file_name).stem.replace("_", " ")


def detect_source_filter(query: str) -> Optional[str]:
    q = str(query).lower()
    if any(x in q for x in ["anayasa", "anayasanın", "anayasa'nın", "geçici madde", "gecici madde"]):
        return "Türkiye Cumhuriyeti Anayasası"
    source_query_rules = [
        ("Türk Ceza Kanunu", ["tck", "türk ceza kanunu", "ceza kanunu"]),
        ("Ceza Muhakemesi Kanunu", ["cmk", "ceza muhakemesi"]),
        ("Türk Medeni Kanunu", ["tmk", "medeni kanun", "miras", "gaiplik"]),
        ("Türk Borçlar Kanunu", ["tbk", "borçlar kanunu", "kira", "kiracı", "sözleşme"]),
        ("Kişisel Verilerin Korunması Kanunu", ["kvkk", "kişisel veri", "ilgili kişi"]),
        ("Bilgi Edinme Hakkı Kanunu", ["bilgi edinme"]),
        ("Hukuk Muhakemeleri Kanunu", ["hmk", "hukuk muhakemeleri"]),
        ("İdari Yargılama Usulü Kanunu", ["iyuk", "idari yargılama"]),
    ]
    for source_name, keywords in source_query_rules:
        if any(keyword in q for keyword in keywords):
            return source_name
    return None


def source_matches(source_filter: str, row: Dict[str, object]) -> bool:
    target = str(source_filter).lower()
    fields = [str(row.get("source", "")), str(row.get("file_name", "")), str(row.get("chunk_text", ""))[:3000]]
    combined = "\n".join(fields).lower()
    if target in combined:
        return True
    alias_map = {
        "türkiye cumhuriyeti anayasası": ["anayasa", "constitution"],
        "türk ceza kanunu": ["tck", "ceza kanunu"],
        "ceza muhakemesi kanunu": ["cmk"],
        "türk medeni kanunu": ["tmk", "medeni kanun"],
        "türk borçlar kanunu": ["tbk", "borçlar kanunu"],
        "kişisel verilerin korunması kanunu": ["kvkk", "kişisel veri"],
        "bilgi edinme hakkı kanunu": ["bilgi edinme"],
    }
    return any(alias in combined for alias in alias_map.get(target, []))


def extract_article_reference(query: str) -> Optional[Dict[str, object]]:
    q = str(query).lower()
    is_temporary = "geçici madde" in q or "gecici madde" in q
    patterns = [
        r"geçici\s+madde\s+(\d+)", r"gecici\s+madde\s+(\d+)", r"madde\s*(\d+)",
        r"(\d+)\.\s*madde", r"(\d+)\s*inci\s*madde", r"(\d+)\s*ıncı\s*madde",
        r"(\d+)\s*uncu\s*madde", r"(\d+)\s*üncü\s*madde", r"(\d+)\.\s*maddesi",
        r"(\d+)\s*maddesi", r"\b(tck|cmk|tmk|tbk|hmk|kvkk)\s*(\d+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            number = match.group(2) if len(match.groups()) >= 2 and match.group(2) else match.group(1)
            return {"article_number": number, "is_temporary": is_temporary}
    return None


def article_match_bonus(query: str, chunk_text: str, chunk_article_number: str = "") -> float:
    ref = extract_article_reference(query)
    if ref is None:
        return 0.0
    article_no = str(ref["article_number"])
    is_temporary = bool(ref["is_temporary"])
    if chunk_article_number and str(chunk_article_number).strip() == article_no:
        return 1.0
    text = str(chunk_text).lower()
    text = re.sub(r"\[[^\]]+\]", " ", text)
    if is_temporary:
        patterns = [
            rf"geçici\s+madde\s+{article_no}\b", rf"gecici\s+madde\s+{article_no}\b",
            rf"geçici\s+{article_no}\.\s*madde", rf"gecici\s+{article_no}\.\s*madde",
        ]
    else:
        patterns = [
            rf"madde\s+{article_no}\b", rf"madde\s*{article_no}\s*[–-]", rf"{article_no}\.\s*madde",
            rf"{article_no}\s*maddesi", rf"{article_no}\s*(inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü)\s+madde",
        ]
    return 1.0 if any(re.search(pattern, text) for pattern in patterns) else 0.0


def extract_article_no_from_text(text: str) -> str:
    head = str(text)[:400].lower()
    patterns = [r"geçici\s+madde\s+(\d+)", r"gecici\s+madde\s+(\d+)", r"madde\s+([0-9]+)\s*[–\-:]", r"madde\s+([0-9]+)\b", r"^\s*([0-9]+)\.\s*madde"]
    for pattern in patterns:
        match = re.search(pattern, head, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1)
    return ""


def character_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
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


def split_article_blocks(text: str) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    pattern = re.compile(r"(?im)(?=^\s*(?:GEÇİCİ\s+MADDE|GECICI\s+MADDE|MADDE)\s+\d+\s*[–\-:]?)")
    parts = [p.strip() for p in pattern.split(text) if p.strip()]
    if len(parts) <= 1:
        return character_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    chunks: List[str] = []
    for part in parts:
        if len(part) <= CHUNK_SIZE * 1.5:
            chunks.append(part)
        else:
            chunks.extend(character_chunks(part, CHUNK_SIZE, CHUNK_OVERLAP))
    return chunks


def build_documents_from_uploads(uploaded_files: Optional[List[UploadFile]], include_repo_docs: bool) -> List[Dict[str, str]]:
    documents: List[Dict[str, str]] = []
    if include_repo_docs and DEFAULT_DOCS_DIR.exists():
        for path in sorted(DEFAULT_DOCS_DIR.rglob("*")):
            if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in SUPPORTED_UPLOAD_TYPES:
                text = read_file_path(path).strip()
                if text:
                    documents.append({"doc_id": path.stem.replace(" ", "_"), "file_name": path.name, "source": infer_source_name(path.name, text), "text": text})
    for uploaded in uploaded_files or []:
        data = uploaded.file.read()
        text = read_file_bytes(uploaded.filename, data).strip()
        if text:
            safe_stem = Path(uploaded.filename).stem.replace(" ", "_")
            documents.append({"doc_id": f"uploaded_{safe_stem}", "file_name": uploaded.filename, "source": infer_source_name(uploaded.filename, text), "text": text})
    if not documents:
        raise ValueError("Hiç doküman bulunamadı. Lütfen .txt, .pdf veya .docx hukuk dokümanı yükleyin.")
    return documents


def build_chunks_from_documents(documents: List[Dict[str, str]]) -> pd.DataFrame:
    rows = []
    for doc in documents:
        blocks = split_article_blocks(doc["text"])
        for idx, block in enumerate(blocks):
            if not block.strip():
                continue
            article_no = extract_article_no_from_text(block)
            rows.append({"chunk_id": f"{doc['doc_id']}_chunk_{idx:04d}", "doc_id": doc["doc_id"], "file_name": doc["file_name"], "source": doc["source"], "article_no": article_no, "chunk_text": block})
    if not rows:
        raise ValueError("Dokümanlar yüklendi ama chunk oluşturulamadı.")
    return pd.DataFrame(rows)


def get_embedding_model() -> SentenceTransformer:
    if "embedding_model" not in _CACHE:
        _CACHE["embedding_model"] = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _CACHE["embedding_model"]


def get_reranker() -> CrossEncoder:
    if "reranker" not in _CACHE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _CACHE["reranker"] = CrossEncoder(TURKISH_BGE_RERANKER_NAME, device=device, max_length=512)
    return _CACHE["reranker"]


def get_mistral(hf_token: str = ""):
    cache_key = "base_mistral"
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    token_arg = hf_token.strip() or os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(BASE_MISTRAL_MODEL_NAME, use_fast=True, token=token_arg)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    model_kwargs = {"low_cpu_mem_usage": True, "token": token_arg}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    else:
        model_kwargs["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE_MISTRAL_MODEL_NAME, **model_kwargs)
    model.eval()
    _CACHE[cache_key] = (tokenizer, model)
    return tokenizer, model


def build_retrieval_index(chunks_df: pd.DataFrame):
    embedding_model = get_embedding_model()
    texts = chunks_df["chunk_text"].astype(str).tolist()
    embeddings = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    tokenized_corpus = [simple_turkish_tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    return embedding_model, embeddings, bm25


def retrieve_candidates_article_source_aware(query: str, chunks_df: pd.DataFrame, embedding_model: SentenceTransformer, embeddings: np.ndarray, bm25: BM25Okapi, candidate_k: int = CANDIDATE_K) -> List[Dict[str, object]]:
    source_filter = detect_source_filter(query)
    candidate_df = chunks_df.copy().reset_index(drop=True)
    candidate_embeddings = embeddings
    used_source_filter = None
    if source_filter:
        mask = candidate_df.apply(lambda row: source_matches(source_filter, row), axis=1)
        filtered_df = candidate_df[mask].reset_index(drop=True)
        if len(filtered_df) >= MIN_FILTERED_ROWS:
            original_indices = candidate_df[mask].index.to_numpy()
            candidate_df = filtered_df
            candidate_embeddings = embeddings[original_indices]
            used_source_filter = source_filter
    candidate_texts = candidate_df["chunk_text"].astype(str).tolist()
    candidate_bm25 = BM25Okapi([simple_turkish_tokenize(text) for text in candidate_texts])
    query_embedding = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    dense_scores = cosine_similarity(query_embedding, candidate_embeddings)[0]
    bm25_scores = np.asarray(candidate_bm25.get_scores(simple_turkish_tokenize(query)), dtype=np.float32)
    dense_norm = min_max_normalize(dense_scores)
    bm25_norm = min_max_normalize(bm25_scores)
    hybrid_scores = ALPHA * dense_norm + (1.0 - ALPHA) * bm25_norm
    article_bonuses = np.asarray([article_match_bonus(query, candidate_df.iloc[i]["chunk_text"], candidate_df.iloc[i].get("article_no", "")) for i in range(len(candidate_df))], dtype=np.float32)
    final_scores = hybrid_scores + ARTICLE_BONUS_WEIGHT * article_bonuses
    top_n = min(candidate_k, len(candidate_df))
    top_indices = np.argsort(final_scores)[::-1][:top_n]
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        row = candidate_df.iloc[int(idx)]
        results.append({"rank": rank, "chunk_id": row["chunk_id"], "file_name": row["file_name"], "source": row["source"], "article_no": row.get("article_no", ""), "score": float(final_scores[int(idx)]), "hybrid_score": float(hybrid_scores[int(idx)]), "dense_score": float(dense_norm[int(idx)]), "bm25_score": float(bm25_norm[int(idx)]), "article_bonus": float(article_bonuses[int(idx)]), "source_filter": used_source_filter, "chunk_text": row["chunk_text"]})
    return results


def rerank_retrieved_chunks_turkish_bge(question: str, retrieved_results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not retrieved_results:
        return []
    reranker = get_reranker()
    pairs = [[question, item["chunk_text"]] for item in retrieved_results]
    scores = reranker.predict(pairs, batch_size=4, show_progress_bar=False)
    scored = []
    for item, score in zip(retrieved_results, scores):
        enriched = dict(item)
        enriched.update({"original_rank": item["rank"], "rerank_score": float(score)})
        scored.append(enriched)
    scored = sorted(scored, key=lambda x: x["rerank_score"], reverse=True)
    for new_rank, item in enumerate(scored, start=1):
        item["rerank_rank"] = new_rank
    return scored


def retrieve_best_pipeline(question: str, chunks_df: pd.DataFrame, embedding_model: SentenceTransformer, embeddings: np.ndarray, bm25: BM25Okapi) -> List[Dict[str, object]]:
    candidates = retrieve_candidates_article_source_aware(question, chunks_df, embedding_model, embeddings, bm25)
    reranked_all = rerank_retrieved_chunks_turkish_bge(question, candidates)
    rerank_rank_map = {item["chunk_id"]: item["rerank_rank"] for item in reranked_all}
    rerank_score_map = {item["chunk_id"]: item["rerank_score"] for item in reranked_all}
    fused_results = []
    for item in candidates:
        original_rank = item["rank"]
        rerank_rank = rerank_rank_map.get(item["chunk_id"], CANDIDATE_K + 1)
        fusion_score = HYBRID_WEIGHT * (1 / original_rank) + RERANK_WEIGHT * (1 / rerank_rank)
        fused = dict(item)
        fused.update({"rank": None, "fusion_score": float(fusion_score), "original_rank": original_rank, "rerank_rank": rerank_rank, "rerank_score": float(rerank_score_map.get(item["chunk_id"], 0.0))})
        fused_results.append(fused)
    fused_results = sorted(fused_results, key=lambda x: x["fusion_score"], reverse=True)
    final = []
    for rank, item in enumerate(fused_results[:FINAL_CONTEXT_K], start=1):
        item["rank"] = rank
        final.append(item)
    return final


def build_improved_legal_rag_prompt(question: str, retrieved_contexts: List[str]) -> str:
    context_text = "\n\n".join([f"[Bağlam {i + 1}]\n{ctx}" for i, ctx in enumerate(retrieved_contexts)])
    prompt = f"""
Sen Türk hukuk metinleri üzerinde çalışan dikkatli bir RAG soru-cevap asistanısın.

Görevin:
Sadece verilen bağlamları kullanarak soruya kısa, net ve hukuki olarak doğru cevap vermek.

Zorunlu kurallar:
1. Cevabı yalnızca verilen bağlamlara dayandır.
2. Bağlamda açıkça bulunmayan bilgiyi uydurma.
3. Soru bir süre, tarih, sayı veya madde soruyorsa sadece ilgili değeri ve kısa açıklamasını ver.
4. Soru "aykırı mıdır", "çelişir mi", "uygun mudur" gibi bir değerlendirme soruyorsa cevaba mutlaka "Evet" veya "Hayır" ile başla.
5. Cevapta "Bağlam", "Context", "verilen metne göre" gibi ifadeler kullanma.
6. Cevap en fazla iki kısa cümle olmalı.
7. Alternatif cevap, yeni soru, örnek soru, başlık veya açıklama bölümü üretme.
8. Cevabı verdikten sonra dur.
9. Eğer bağlamda cevap yoksa sadece şunu yaz: "Verilen bağlamda bu sorunun cevabı bulunamamaktadır."

Bağlamlar:
{context_text}

Soru:
{question}

Kısa ve doğrudan cevap:
"""
    return f"<s>[INST] {prompt.strip()} [/INST]"


def clean_generated_answer(text: str) -> str:
    text = str(text).strip()
    for marker in ["Kısa ve doğrudan cevap:", "Kısa cevap:", "Answer:", "Cevap:"]:
        if marker in text:
            text = text.split(marker)[-1].strip()
    for marker in ["\nBaşka bir şekilde:", "\nCevap:", "\nSoru:", "\nQuestion:", "\nAlternatif", "\nAçıklama:", "\nDetaylı cevap:", "\nÖrnek:", "</s>"]:
        if marker in text:
            text = text.split(marker)[0].strip()
    for u in ["Context 1", "Context 2", "Context 3", "Bağlam 1", "Bağlam 2", "Bağlam 3"]:
        text = text.replace(u, "")
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def postprocess_answer_by_question_type(question: str, answer: str) -> str:
    q = str(question).lower()
    answer = str(answer).strip()
    if any(trigger in q for trigger in ["kaç", "ne zaman", "kime aittir", "en fazla kaç"]):
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        if sentences:
            return sentences[0].strip()
    return answer


def generate_with_base_mistral(prompt: str, hf_token: str = "", max_new_tokens: int = 120) -> str:
    tokenizer, model = get_mistral(hf_token)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    if torch.cuda.is_available():
        inputs = inputs.to(model.device)
    input_length = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    new_tokens = outputs[0][input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class HealthResponse(BaseModel):
    ok: bool
    cuda_available: bool
    gpu_name: str
    model: str


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(ok=True, cuda_available=torch.cuda.is_available(), gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU", model=BASE_MISTRAL_MODEL_NAME)


@app.post("/answer")
async def answer(
    question: str = Form(...),
    include_repo_docs: bool = Form(False),
    hf_token: str = Form(""),
    files: Optional[List[UploadFile]] = File(None),
):
    documents = build_documents_from_uploads(files, include_repo_docs=include_repo_docs)
    chunks_df = build_chunks_from_documents(documents)
    embedding_model, embeddings, bm25 = build_retrieval_index(chunks_df)
    retrieved = retrieve_best_pipeline(question, chunks_df, embedding_model, embeddings, bm25)
    contexts = [item["chunk_text"] for item in retrieved]
    prompt = build_improved_legal_rag_prompt(question, contexts)
    raw_answer = generate_with_base_mistral(prompt, hf_token=hf_token)
    cleaned = postprocess_answer_by_question_type(question, clean_generated_answer(raw_answer))
    safe_sources = []
    for item in retrieved:
        safe_sources.append({
            "rank": item.get("rank"),
            "source": item.get("source"),
            "file_name": item.get("file_name"),
            "article_no": item.get("article_no"),
            "fusion_score": item.get("fusion_score"),
            "rerank_score": item.get("rerank_score"),
            "article_bonus": item.get("article_bonus"),
            "source_filter": item.get("source_filter"),
            "chunk_text": item.get("chunk_text"),
        })
    return {"answer": cleaned, "sources": safe_sources, "num_documents": len(documents), "num_chunks": len(chunks_df)}
