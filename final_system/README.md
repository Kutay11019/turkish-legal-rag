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

* `.txt`
* `.pdf`
* `.docx`

## Modes

The runner supports three modes:

* `base`: runs the RAG system with the base Mistral model.
* `finetuned`: runs the RAG system with the same base Mistral model plus a fine-tuned QLoRA adapter.
* `both`: runs both systems on the same benchmark and saves comparison outputs.

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

This mode can run on a normal CPU environment and is useful for testing custom document loading and retrieval.

## Run Base RAG

```bash
python run_custom_rag_benchmark.py --mode base
```

## Run Base vs Fine-Tuned RAG

Fine-tuned mode requires the QLoRA adapter files.

The adapter is not committed to GitHub because it contains large model weight files.

Download the clean fine-tuned adapter zip from:

```text
https://drive.google.com/drive/folders/1yVp0kwU0ImgiG3dnE6z-QHI3ZcIaHovQ?usp=sharing
```

After downloading the zip file, create this folder if it does not already exist:

```text
final_system/models/mistral_legal_qlora_starlar_v2_800steps/
```

Then extract the contents of the zip file into that folder.

The folder should contain files such as:

```text
adapter_config.json
adapter_model.safetensors
tokenizer_config.json
tokenizer.json
chat_template.jinja
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

## Hardware Requirement

The retrieval-only mode can run on a normal CPU environment:

```bash
python run_custom_rag_benchmark.py --mode base --retrieval_only
```

However, full answer generation with Mistral-7B requires sufficient GPU memory. For this reason, running `--mode base`, `--mode finetuned`, or `--mode both` is recommended on a CUDA-enabled GPU environment such as Google Colab L4/A100 or a local machine with enough VRAM.

If the system is executed on a CPU-only machine, the retrieval pipeline can still be tested with `--retrieval_only`, but full LLM generation may be very slow or may fail due to memory limitations.

## Notes

The `--retrieval_only` option can be used to verify that the custom document and benchmark pipeline works without loading the LLM.

Full base/fine-tuned RAG comparison requires:

1. Sufficient GPU memory.
2. Access to the base Mistral model.
3. The fine-tuned QLoRA adapter placed under `final_system/models/`.
4. `local_finetuned_adapter` configured correctly in `config.yaml`.
