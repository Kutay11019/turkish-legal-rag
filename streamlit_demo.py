from __future__ import annotations

from typing import List

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
st.caption("Upload legal documents, ask a question, get one generated answer.")

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

st.subheader("1. Upload Legal Document(s)")
uploaded_files = st.file_uploader(
    "Upload .txt, .pdf, or .docx legal document(s)",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True,
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
        st.error("Lütfen en az bir doküman yükle veya repo sample documents seçeneğini aç.")
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
                timeout=600,
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
