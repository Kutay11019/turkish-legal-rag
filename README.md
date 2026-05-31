# CENG493 Turkish Legal RAG

This repository contains a Turkish legal question answering system based on an optimized Retrieval-Augmented Generation (RAG) pipeline.

The project includes retrieval improvements, reranking experiments, LLM fine-tuning with QLoRA, controlled evaluation, ablation studies, and a final runnable custom RAG benchmark system.

## Project Overview

The goal of this project is to build and evaluate a Turkish legal RAG system that can answer legal questions by retrieving relevant legal contexts and generating source-grounded answers.

The project focuses on:

* Turkish legal question answering
* Retrieval-Augmented Generation (RAG)
* Hybrid retrieval
* Turkish BGE reranking
* Source-aware and article-aware retrieval improvements
* Prompt engineering for legal answer generation
* LLM fine-tuning with QLoRA
* Base vs fine-tuned LLM comparison
* Custom document and benchmark evaluation support

## Final Runnable System

The final runnable system is located under:

```text
final_system/
```

This folder provides a clean interface for running the RAG system on custom legal documents and custom benchmark question-answer sets.

The instructor can place custom legal documents under:

```text
final_system/data/custom_documents/
```

and a custom benchmark CSV under:

```text
final_system/data/custom_benchmark/
```

The benchmark CSV must include at least:

```csv
question,expected_answer
```

The runner supports three modes:

* `base`: runs the RAG system with the base Mistral model.
* `finetuned`: runs the RAG system with the same base Mistral model plus the fine-tuned QLoRA adapter.
* `both`: runs both systems on the same benchmark and saves comparison outputs.

For detailed usage instructions, see:

```text
final_system/README.md
```

## Quick Smoke Test

A retrieval-only test can be run without loading the LLM:

```bash
cd final_system
python run_custom_rag_benchmark.py --mode base --retrieval_only
```

This checks whether:

* custom documents can be loaded,
* documents can be chunked,
* benchmark questions can be read,
* retrieval works,
* output CSV files are generated.

This mode can run on a normal CPU environment.

## Full LLM Generation

Full answer generation with Mistral-7B requires sufficient GPU memory.

For this reason, running the following modes is recommended on a CUDA-enabled GPU environment such as Google Colab L4/A100 or a local machine with enough VRAM:

```bash
python run_custom_rag_benchmark.py --mode base
python run_custom_rag_benchmark.py --mode finetuned
python run_custom_rag_benchmark.py --mode both
```

If local GPU resources are not available, the Colab runner can be used:

```text
notebooks/27_final_system_colab_runner.ipynb
```

Recommended Colab runtime:

```text
Runtime > Change runtime type > L4 GPU
```

The Colab runner clones the repository, installs the final system dependencies, downloads the fine-tuned adapter, updates the adapter path, and runs the final RAG system.

## Fine-Tuned Adapter

The fine-tuned QLoRA adapter is not committed to GitHub because it contains large model weight files.

The clean adapter zip is linked inside:

```text
final_system/README.md
```

After downloading the adapter, extract it under:

```text
final_system/models/mistral_legal_qlora_starlar_v2_800steps/
```

Then update `final_system/config.yaml`:

```yaml
local_finetuned_adapter: "models/mistral_legal_qlora_starlar_v2_800steps"
```

Then run:

```bash
cd final_system
python run_custom_rag_benchmark.py --mode both
```

## Final Experimental Results

The final experiment summary is available at:

```text
reports/final_experiment_summary.md
```

### Best End-to-End RAG Result

The best completed end-to-end RAG pipeline on the original benchmark used:

* Source-aware retrieval
* Article-aware retrieval
* Turkish BGE reranker
* Improved legal prompt
* Base Mistral generator

| Setting                  | Manual Accuracy |
| ------------------------ | --------------: |
| Original benchmark       |           0.421 |
| Coverage-clean benchmark |         0.46875 |

The coverage-clean score excludes known problematic samples caused by missing or mismatched corpus coverage.

### LLM Fine-Tuning Result

LLM fine-tuning was evaluated separately under a controlled gold-context setting using the Starlar LLM SFT dataset.

In this setting, the correct context was already provided to the model, so the experiment focused on source-grounded answer generation.

| Model                            | Manual Accuracy | Mean Token F1 | Mean Text Similarity | Source Citation Rate |
| -------------------------------- | --------------: | ------------: | -------------------: | -------------------: |
| Base Mistral                     |           0.400 |        0.1848 |               0.1554 |                 0.25 |
| Starlar Fine-tuned Mistral QLoRA |           0.925 |        0.9657 |               0.9695 |                 0.85 |

These results show that LLM fine-tuning substantially improved source-grounded Turkish legal answer generation when the correct context was provided.

### Integration and Ablation Notes

The Starlar fine-tuned LLM was also tested on the fixed retrieved contexts from the previous best RAG pipeline. However, it did not improve the original RAG benchmark result because of dataset distribution differences, answer-format mismatch, and retrieval/context coverage limitations.

A target-style alignment fine-tuning experiment was also attempted, but qualitative sanity checks showed generic or legally inaccurate outputs. Therefore, this adapter was not selected as the final RAG generator.

The final project reports two complementary findings:

1. Retrieval and prompt engineering produced the best end-to-end RAG result on the original benchmark.
2. LLM fine-tuning clearly improved controlled source-grounded answer generation on the Starlar dataset.

## Repository Structure

The repository is organized as follows:

```text
.
├── final_system/
│   ├── README.md
│   ├── requirements.txt
│   ├── config.yaml
│   ├── run_custom_rag_benchmark.py
│   │
│   ├── data/
│   │   ├── custom_documents/
│   │   │   ├── README.md
│   │   │   └── sample_legal_document.txt
│   │   │
│   │   ├── custom_benchmark/
│   │   │   └── sample_benchmark.csv
│   │   │
│   │   └── outputs/
│   │       └── .gitkeep
│   │
│   ├── models/
│   │   └── .gitkeep
│   │
│   └── src/
│       ├── __init__.py
│       ├── chunking.py
│       ├── document_loader.py
│       ├── generation.py
│       ├── metrics.py
│       ├── retrieval.py
│       └── utils.py
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_baseline_retrieval.ipynb
│   ├── 03_rag_generation.ipynb
│   ├── 04_flashrank_reranker_experiment.ipynb
│   ├── 05_turkish_bge_reranker_experiment.ipynb
│   ├── 06_error_analysis.ipynb
│   ├── 07_prompt_context_improvement.ipynb
│   ├── 08_article_aware_retrieval_improvement.ipynb
│   ├── 09_data_coverage_analysis.ipynb
│   ├── 10_multi_dataset_smoke_test.ipynb
│   ├── 11_reranker_finetuning.ipynb
│   ├── 12_llm_finetuning_data_preparation.ipynb
│   ├── 13_llm_finetuning_qlora_friend_clean.ipynb
│   ├── 14_evaluate_finetuned_llm_rag.ipynb
│   ├── 18_starlar_llm_finetune_v2_inspection.ipynb
│   ├── 19_llm_finetune_starlar_v2.ipynb
│   ├── 20_evaluate_starlar_finetuned_llm.ipynb
│   ├── 21_final_results_summary.ipynb
│   ├── 22_evaluate_starlar_finetuned_llm_on_old_best_rag_contexts.ipynb
│   ├── 23_prepare_target_style_sft_for_rag_alignment.ipynb
│   ├── 24_finetune_starlar_adapter_target_style_alignment.ipynb
│   └── 27_final_system_colab_runner.ipynb
│
├── outputs/
│   └── metrics/
│       └── selected final metric CSV files
│
├── reports/
│   └── final_experiment_summary.md
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Important Files

| Path                                                      | Purpose                                                                                                             |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `final_system/`                                           | Final runnable custom RAG benchmark system. This is the main folder to use for custom evaluation.                   |
| `final_system/README.md`                                  | Detailed instructions for running the final system, using custom documents, and configuring the fine-tuned adapter. |
| `final_system/run_custom_rag_benchmark.py`                | Main script for running `base`, `finetuned`, or `both` RAG modes.                                                   |
| `final_system/config.yaml`                                | Configuration file for model names, adapter path, retrieval settings, generation settings, and input/output paths.  |
| `final_system/data/custom_documents/`                     | Folder where custom `.txt`, `.pdf`, or `.docx` legal documents should be placed.                                    |
| `final_system/data/custom_benchmark/sample_benchmark.csv` | Example benchmark format. A custom benchmark must include `question` and `expected_answer` columns.                 |
| `final_system/src/`                                       | Source code for document loading, chunking, retrieval, generation, metrics, and utility functions.                  |
| `notebooks/27_final_system_colab_runner.ipynb`            | Colab runner for GPU-based testing when local hardware is not sufficient for Mistral-7B generation.                 |
| `reports/final_experiment_summary.md`                     | Final experiment summary, including best RAG result, LLM fine-tuning result, and ablation conclusions.              |
| `outputs/metrics/`                                        | Selected final metric CSV files supporting the reported results.                                                    |
| `notebooks/`                                              | Experimental notebooks documenting the research and development process.                                            |


## Selected Metric Outputs

Selected final metric outputs are stored under:

```text
outputs/metrics/
```

These files support the reported results, including:

* best end-to-end RAG performance,
* retrieval and reranking comparisons,
* Starlar LLM fine-tuning results,
* controlled base vs fine-tuned evaluation,
* integration and target-alignment ablation results.

Intermediate and large generated files are excluded from GitHub where appropriate.

## Dataset Notes

Large raw and processed datasets are not committed to GitHub because of size limitations.

The project originally used Turkish legal QA and legal text resources, including external Turkish legal datasets and the Starlar LLM SFT dataset prepared during the experiments.

For custom evaluation, users do not need the original training datasets. They can place their own legal documents and benchmark CSV files under `final_system/`.

## Installation for Final System

From the project root:

```bash
cd final_system
pip install -r requirements.txt
```

Then run the retrieval-only smoke test:

```bash
python run_custom_rag_benchmark.py --mode base --retrieval_only
```

For full base/fine-tuned generation, use a CUDA-enabled GPU environment.

## Notes About Sample Benchmark

The included sample files under `final_system/data/` are only smoke-test examples.

They are used to verify that the final runner can load documents, retrieve contexts, run modes, and save outputs.

The scores produced on the two-question sample benchmark are not the final project performance scores.

The final reported results are documented in:

```text
reports/final_experiment_summary.md
```
