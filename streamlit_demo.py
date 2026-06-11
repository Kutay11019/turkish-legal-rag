from __future__ import annotations

from io import StringIO
from typing import List

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Turkish Legal RAG", page_icon="⚖️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 980px;}
    div[data-testid="stExpander"] {border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚖️ Turkish Legal RAG")
st.caption("Upload legal files, ask a question, get one generated answer.")

with st.sidebar:
    st.subheader("Backend")
    backend_url = st.text_input(
        "Colab GPU backend URL",
        value=st.session_state.get("backend_url", ""),
        placeholder="https://xxxx.trycloudflare.com",
        help="Colab'da çalışan FastAPI backend linkini buraya yapıştır.",
    ).strip().rstrip("/")
    if backend_url:
        st.session_state["backend_url"] = backend_url

    hf_token = st.text_input(
        "Hugging Face token (optional)",
        type="password",
        help="Mistral model erişimi gerekirse kullanılır. Colab tarafında HF_TOKEN ayarlıysa boş bırakabilirsin.",
    )

    include_repo_docs = st.checkbox(
        "Also include repo sample documents",
        value=False,
        help="Açık olursa backend'deki final_system/data/custom_documents da kullanılır. Hoca kendi dosyasını yükleyince genelde kapalı kalsın.",
    )

    if st.button("Check backend", use_container_width=True):
        if not backend_url:
            st.warning("Önce backend URL gir.")
        else:
            try:
                r = requests.get(f"{backend_url}/health", timeout=20)
                r.raise_for_status()
                data = r.json()
                st.success(f"Backend OK | GPU: {data.get('gpu_name')}")
            except Exception as exc:
                st.error(f"Backend'e ulaşılamadı: {exc}")

st.subheader("1. Upload Legal File(s)")
uploaded_files = st.file_uploader(
    "Upload .txt, .pdf, .docx, .csv, .json, or .jsonl file(s)",
    type=["txt", "pdf", "docx", "csv", "json", "jsonl"],
    accept_multiple_files=True,
)

st.info(
    "CSV/JSON/JSONL dosyaları iki şekilde kullanılabilir: Eğer içinde hukuk metni varsa yukarıya legal file olarak yükle. "
    "Eğer içinde question/expected_answer satırları varsa aşağıdaki Batch Benchmark alanına yükle."
)

st.subheader("2. Ask a Question")
question = st.text_area(
    "Question",
    value="Anayasa 10. madde neyi düzenler?",
    height=120,
    label_visibility="collapsed",
)

generate = st.button("Generate Answer", type="primary", use_container_width=True)

if generate:
    if not backend_url:
        st.error("Önce sol taraftaki Colab GPU backend URL alanına link yapıştır.")
        st.stop()
    if not question.strip():
        st.error("Lütfen bir soru yaz.")
        st.stop()
    if not uploaded_files and not include_repo_docs:
        st.error("Lütfen en az bir dosya yükle veya repo sample documents seçeneğini aç.")
        st.stop()

    files_payload: List[tuple] = []
    for uploaded in uploaded_files:
        files_payload.append(("files", (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")))

    data = {
        "question": question,
        "include_repo_docs": str(include_repo_docs).lower(),
        "hf_token": hf_token or "",
    }

    try:
        with st.spinner("Generating answer on Colab GPU backend..."):
            response = requests.post(
                f"{backend_url}/answer",
                data=data,
                files=files_payload,
                timeout=900,
            )
            response.raise_for_status()
            result = response.json()

        st.subheader("Generated Answer")
        st.write(result.get("answer") or "Cevap üretilemedi.")

        st.caption(f"Documents: {result.get('num_documents')} | Chunks: {result.get('num_chunks')}")

        with st.expander("Retrieved Legal Sources", expanded=False):
            for item in result.get("sources", []):
                st.markdown(f"**Rank {item.get('rank')} — {item.get('source')} / {item.get('file_name')}**")
                if item.get("article_no"):
                    st.caption(f"Article: {item.get('article_no')}")
                st.caption(
                    f"fusion={float(item.get('fusion_score') or 0):.4f} | "
                    f"rerank={float(item.get('rerank_score') or 0):.4f} | "
                    f"article_bonus={float(item.get('article_bonus') or 0):.2f} | "
                    f"source_filter={item.get('source_filter') or '-'}"
                )
                st.write(item.get("chunk_text"))
                st.markdown("---")

    except requests.exceptions.Timeout:
        st.error("Backend zaman aşımına uğradı. İlk çalıştırmada model yükleniyor olabilir; biraz bekleyip tekrar deneyebilirsin.")
    except Exception as exc:
        st.error("Answer generation failed.")
        st.exception(exc)

st.markdown("---")

with st.expander("Optional: Batch Benchmark Test (.csv / .json / .jsonl)", expanded=False):
    st.write(
        "Buraya question/soru ve isteğe bağlı expected_answer/cevap kolonları olan benchmark dosyası yükleyebilirsin. "
        "Sistem aynı legal files üzerinden soruları sırayla cevaplar ve sonuçları indirilebilir CSV olarak verir."
    )

    benchmark_file = st.file_uploader(
        "Upload benchmark file",
        type=["csv", "json", "jsonl"],
        accept_multiple_files=False,
        key="benchmark_uploader",
    )

    max_questions = st.number_input(
        "Max questions to run",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        help="Mistral generation yavaş olabileceği için demo sırasında 3-5 soru idealdir.",
    )

    run_benchmark = st.button("Run Benchmark", use_container_width=True)

    if run_benchmark:
        if not backend_url:
            st.error("Önce sol taraftaki Colab GPU backend URL alanına link yapıştır.")
            st.stop()
        if benchmark_file is None:
            st.error("Lütfen .csv, .json veya .jsonl benchmark dosyası yükle.")
            st.stop()
        if not uploaded_files and not include_repo_docs:
            st.error("Benchmark çalıştırmak için en az bir legal file yükle veya repo sample documents seçeneğini aç.")
            st.stop()

        files_payload: List[tuple] = []
        for uploaded in uploaded_files:
            files_payload.append(("files", (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")))
        files_payload.append(("benchmark_file", (benchmark_file.name, benchmark_file.getvalue(), benchmark_file.type or "application/octet-stream")))

        data = {
            "include_repo_docs": str(include_repo_docs).lower(),
            "hf_token": hf_token or "",
            "max_questions": str(int(max_questions)),
        }

        try:
            with st.spinner("Running benchmark on Colab GPU backend..."):
                response = requests.post(
                    f"{backend_url}/benchmark",
                    data=data,
                    files=files_payload,
                    timeout=1800,
                )
                response.raise_for_status()
                result = response.json()

            rows = result.get("results", [])
            if not rows:
                st.warning("Benchmark sonucu boş geldi.")
            else:
                df = pd.DataFrame(rows)
                st.success(
                    f"Benchmark completed. Questions: {result.get('num_questions')} | "
                    f"Documents: {result.get('num_documents')} | Chunks: {result.get('num_chunks')}"
                )
                st.dataframe(df, use_container_width=True)

                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
                st.download_button(
                    "Download results CSV",
                    data=csv_buffer.getvalue().encode("utf-8-sig"),
                    file_name="rag_benchmark_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except requests.exceptions.Timeout:
            st.error("Benchmark zaman aşımına uğradı. Max questions değerini düşürüp tekrar deneyebilirsin.")
        except Exception as exc:
            st.error("Benchmark failed.")
            st.exception(exc)
