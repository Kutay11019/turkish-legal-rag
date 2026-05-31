# Final Custom RAG Benchmark Runner

This folder contains the final runnable RAG system for custom document collections and custom benchmark question-answer sets.

## Purpose

The instructor can place custom legal documents under:

```text
final_system/data/custom_documents/
```

and a custom benchmark CSV under:

```text
final_system/data/custom_benchmark/
```

The benchmark CSV must include at least the following columns:

```csv
question,expected_answer
```

## Supported Document Formats

The document loader supports:

- `.txt`
- `.pdf`
- `.docx`

## Modes

The runner supports three modes:

- `base`: runs the RAG system with the base Mistral model.
- `finetuned`: runs the RAG system with the same base Mistral model plus a fine-tuned QLoRA adapter.
- `both`: runs both systems on the same benchmark and saves comparison outputs.

## Install

From the project root:

```bash
cd final_system
pip install -r requirements.txt
```

## Quick Retrieval-Only Test

This checks document loading, chunking, retrieval, and output writing without loading the LLM:

```bash
python run_custom_rag_benchmark.py --mode base --retrieval_only
```

## Run Base RAG

```bash
python run_custom_rag_benchmark.py --mode base
```

## Run Base vs Fine-Tuned RAG

Fine-tuned mode requires the QLoRA adapter files.

The adapter is not committed to GitHub because it contains large model weight files.

After downloading the adapter, place it under:

```text
final_system/models/mistral_legal_qlora_starlar_v2_800steps/
```

Then update `config.yaml`:

```yaml
local_finetuned_adapter: "models/mistral_legal_qlora_starlar_v2_800steps"
```

Then run:

```bash
python run_custom_rag_benchmark.py --mode both
```

## Outputs

Results are saved under:

```text
final_system/data/outputs/
```

Expected output files:

```text
base_rag_results.csv
finetuned_rag_results.csv
base_vs_finetuned_summary.csv
```

## Notes

Running Mistral-based generation requires sufficient GPU memory. The `--retrieval_only` option can be used to test the custom document and benchmark pipeline without loading the LLM.