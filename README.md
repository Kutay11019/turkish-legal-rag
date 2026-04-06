# CENG493 Turkish Legal RAG

This project aims to develop a Turkish legal question answering system using an optimized Retrieval-Augmented Generation (RAG) pipeline.

## Planned System Components
- LLM: Llama
- Fine-tuning: Supervised Instruction Fine-Tuning with QLoRA
- Retrieval Backend: FAISS
- Reranking: Yes
- Evaluation: Retrieval, QA, hallucination, and citation analysis

## Project Structure
- `data/` → raw, processed, and evaluation datasets
- `notebooks/` → exploratory analysis notebooks
- `src/` → source code for preprocessing, retrieval, reranking, training, and evaluation
- `reports/` → progress and final reports
- `outputs/` → generated outputs and logs