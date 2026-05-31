from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

from src.chunking import build_chunks
from src.document_loader import load_documents
from src.generation import RAGGenerator, build_rag_prompt
from src.metrics import has_source_citation, summarize_results, text_similarity, token_f1
from src.retrieval import HybridRetriever
from src.utils import ensure_dir, get_project_root, load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run custom Turkish Legal RAG benchmark."
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML file.",
    )

    parser.add_argument(
        "--documents_dir",
        default=None,
        help="Directory containing custom documents.",
    )

    parser.add_argument(
        "--benchmark_path",
        default=None,
        help="CSV file containing question and expected_answer columns.",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for output CSV files.",
    )

    parser.add_argument(
        "--mode",
        choices=["base", "finetuned", "both"],
        default="both",
        help="Which system to run.",
    )

    parser.add_argument(
        "--finetuned_adapter",
        default=None,
        help="Hugging Face adapter repo or local adapter path.",
    )

    parser.add_argument(
        "--retrieval_only",
        action="store_true",
        help="Only retrieve contexts without running LLM generation.",
    )

    return parser.parse_args()


def load_benchmark(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    df = pd.read_csv(path)

    required_columns = ["question", "expected_answer"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Benchmark file must include columns {required_columns}. "
            f"Missing: {missing}"
        )

    df = df.dropna(subset=["question", "expected_answer"]).copy()
    df["question"] = df["question"].astype(str)
    df["expected_answer"] = df["expected_answer"].astype(str)

    if df.empty:
        raise ValueError("Benchmark file has no valid rows.")

    return df.reset_index(drop=True)


def serialize_contexts(contexts: List[Dict[str, object]]) -> str:
    serializable = []

    for ctx in contexts:
        serializable.append(
            {
                "rank": ctx["rank"],
                "chunk_id": ctx["chunk_id"],
                "file_name": ctx["file_name"],
                "text": ctx["text"],
                "dense_score": ctx["dense_score"],
                "bm25_score": ctx["bm25_score"],
                "hybrid_score": ctx["hybrid_score"],
            }
        )

    return json.dumps(serializable, ensure_ascii=False)


def run_single_mode(
    mode_name: str,
    benchmark_df: pd.DataFrame,
    retriever: HybridRetriever,
    config: Dict[str, object],
    output_dir: Path,
    adapter_id_or_path: Optional[str] = None,
    retrieval_only: bool = False,
) -> pd.DataFrame:
    retrieval_cfg = config["retrieval"]
    generation_cfg = config["generation"]
    model_cfg = config["models"]

    generator = None

    if not retrieval_only:
        generator = RAGGenerator(
            base_model_name=model_cfg["base_llm"],
            adapter_id_or_path=adapter_id_or_path,
            max_input_length=int(generation_cfg["max_input_length"]),
            max_new_tokens=int(generation_cfg["max_new_tokens"]),
            temperature=float(generation_cfg["temperature"]),
            do_sample=bool(generation_cfg["do_sample"]),
            use_4bit=True,
        )

    rows = []

    for idx, row in tqdm(
        benchmark_df.iterrows(),
        total=len(benchmark_df),
        desc=f"Running {mode_name}",
    ):
        question = str(row["question"])
        expected_answer = str(row["expected_answer"])

        contexts = retriever.retrieve(
            query=question,
            top_k=int(retrieval_cfg["top_k"]),
            candidate_k=int(retrieval_cfg["candidate_k"]),
        )

        prompt = build_rag_prompt(question, contexts)

        if retrieval_only:
            generated_answer = ""
        else:
            generated_answer = generator.generate(prompt)

        result_row = {
            "index": idx,
            "mode": mode_name,
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "retrieved_contexts": serialize_contexts(contexts),
            "top1_chunk_id": contexts[0]["chunk_id"] if contexts else "",
            "top1_file_name": contexts[0]["file_name"] if contexts else "",
            "top1_context": contexts[0]["text"] if contexts else "",
            "token_f1": token_f1(generated_answer, expected_answer)
            if generated_answer
            else 0.0,
            "text_similarity": text_similarity(generated_answer, expected_answer)
            if generated_answer
            else 0.0,
            "has_source_citation": has_source_citation(generated_answer)
            if generated_answer
            else 0,
        }

        rows.append(result_row)

    results_df = pd.DataFrame(rows)

    output_path = output_dir / f"{mode_name}_rag_results.csv"
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved {mode_name} results to: {output_path}")

    del generator
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results_df


def make_summary(results_by_mode: Dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    summary_rows = []

    for mode_name, df in results_by_mode.items():
        rows = df.to_dict(orient="records")
        summary = summarize_results(rows)
        summary["mode"] = mode_name
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)

    cols = ["mode", "sample_count", "mean_token_f1", "mean_text_similarity", "source_citation_rate"]
    summary_df = summary_df[cols]

    output_path = output_dir / "base_vs_finetuned_summary.csv"
    summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved summary to: {output_path}")

    return summary_df


def main() -> None:
    args = parse_args()

    final_system_root = get_project_root()
    config_path = resolve_path(args.config, final_system_root)
    config = load_config(config_path)

    paths_cfg = config["paths"]
    model_cfg = config["models"]
    retrieval_cfg = config["retrieval"]

    documents_dir = resolve_path(
        args.documents_dir or paths_cfg["documents_dir"],
        final_system_root,
    )

    benchmark_path = resolve_path(
        args.benchmark_path or paths_cfg["benchmark_path"],
        final_system_root,
    )

    output_dir = resolve_path(
        args.output_dir or paths_cfg["outputs_dir"],
        final_system_root,
    )

    ensure_dir(output_dir)

    print("=" * 80)
    print("Turkish Legal RAG Final System")
    print("=" * 80)
    print("Documents directory:", documents_dir)
    print("Benchmark path:", benchmark_path)
    print("Output directory:", output_dir)
    print("Mode:", args.mode)
    print("Retrieval only:", args.retrieval_only)
    print("=" * 80)

    documents = load_documents(documents_dir)
    print(f"Loaded documents: {len(documents)}")

    chunks = build_chunks(
        documents,
        chunk_size=int(retrieval_cfg["chunk_size"]),
        chunk_overlap=int(retrieval_cfg["chunk_overlap"]),
    )
    print(f"Built chunks: {len(chunks)}")

    benchmark_df = load_benchmark(benchmark_path)
    print(f"Benchmark rows: {len(benchmark_df)}")

    retriever = HybridRetriever(
        chunks=chunks,
        embedding_model_name=model_cfg["embedding_model"],
        hybrid_alpha=float(retrieval_cfg["hybrid_alpha"]),
    )

    modes_to_run = ["base", "finetuned"] if args.mode == "both" else [args.mode]
    results_by_mode = {}

    adapter_value = (
        args.finetuned_adapter
        or model_cfg.get("finetuned_adapter")
        or model_cfg.get("local_finetuned_adapter")
        or ""
    )

    for mode_name in modes_to_run:
        adapter_id_or_path = None

        if mode_name == "finetuned":
            if not adapter_value:
                raise ValueError(
                    "Fine-tuned mode requested, but no adapter is configured. "
                    "Set models.finetuned_adapter or models.local_finetuned_adapter in config.yaml, "
                    "or pass --finetuned_adapter."
                )

            adapter_id_or_path = adapter_value

        results_df = run_single_mode(
            mode_name=mode_name,
            benchmark_df=benchmark_df,
            retriever=retriever,
            config=config,
            output_dir=output_dir,
            adapter_id_or_path=adapter_id_or_path,
            retrieval_only=args.retrieval_only,
        )

        results_by_mode[mode_name] = results_df

    summary_df = make_summary(results_by_mode, output_dir)

    print("\nFinal summary:")
    print(summary_df)


if __name__ == "__main__":
    main()