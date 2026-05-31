# Final Custom RAG Benchmark Runner

This folder contains the final runnable RAG system for custom document collections and custom benchmark question-answer sets.

## Purpose

The instructor can place custom legal documents under:

```text
final_system/data/custom_documents/
```

and a custom benchmark file under:

```text
final_system/data/custom_benchmark/
```

The benchmark file can be provided as one of the following formats:

* `.csv`
* `.json`
* `.jsonl`

Each benchmark sample must contain at least:

```text
question
expected_answer
```

By default, the runner reads the benchmark folder defined in `config.yaml`:

```yaml
benchmark_path: "data/custom_benchmark"
```

If only the included `sample_benchmark.csv` file exists, it is used as a small smoke test. If a custom benchmark file is added to the same folder, the runner automatically prefers the custom benchmark file over the sample file.

If multiple custom benchmark files are placed in the folder, either remove the extra files or pass the exact file path manually with `--benchmark_path`.

Example:

```bash
python run_custom_rag_benchmark.py --mode base --retrieval_only --benchmark_path data/custom_benchmark/my_benchmark.json
```

## Supported Document Formats

The document loader supports:

* `.txt`
* `.pdf`
* `.docx`

## Benchmark Format Examples

### CSV Example

```csv
question,expected_answer
"Anayasa 10. madde neyi düzenler?","Anayasa 10. madde, herkesin kanun önünde eşit olduğunu düzenler."
```

### JSON Example

```json
[
  {
    "question": "Anayasa 10. madde neyi düzenler?",
    "expected_answer": "Anayasa 10. madde, herkesin kanun önünde eşit olduğunu düzenler."
  }
]
```

### JSONL Example

```jsonl
{"question": "Anayasa 10. madde neyi düzenler?", "expected_answer": "Anayasa 10. madde, herkesin kanun önünde eşit olduğunu düzenler."}
```

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
https://drive.google.com/file/d/1VyKe4-oydf8LUs_rmnu02079tFCZkIGO/view?usp=sharing
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

## Google Colab Runner

If a local machine does not have enough GPU memory to run Mistral-7B generation, the provided Colab runner can be used instead:

```text
notebooks/27_final_system_colab_runner.ipynb
```

Open directly in Colab:

```text
https://colab.research.google.com/github/Kutay11019/turkish-legal-rag/blob/main/notebooks/27_final_system_colab_runner.ipynb
```

Recommended Colab runtime:

```text
Runtime > Change runtime type > Hardware accelerator: GPU > GPU type: L4
```

The Colab notebook clones the GitHub repository, installs the final system requirements, downloads the fine-tuned adapter, updates the adapter path, and runs the final RAG runner.

Use this notebook for GPU-based testing of:

* retrieval-only mode,
* base RAG mode,
* fine-tuned RAG mode,
* base vs fine-tuned comparison mode.

## Using Custom Documents and Benchmarks

To use custom legal documents, place them under:

```text
final_system/data/custom_documents/
```

To use a custom benchmark, place a `.csv`, `.json`, or `.jsonl` file under:

```text
final_system/data/custom_benchmark/
```

The runner automatically selects the benchmark file as follows:

1. If a custom benchmark file exists, it is preferred over `sample_benchmark.csv`.
2. If only `sample_benchmark.csv` exists, it is used for smoke testing.
3. If multiple custom benchmark files exist, the runner asks the user to provide the exact file path using `--benchmark_path`.

After adding or replacing benchmark/document files, the notebook does not need to be restarted from the beginning. Continue from the next cell after the file upload or replacement step.

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

## Important Note About Sample Benchmark Scores

The included `sample_legal_document.txt` and `sample_benchmark.csv` files are only provided as a small smoke test.

Their purpose is to verify that:

* custom documents can be loaded,
* benchmark questions can be read,
* retrieval works,
* base and fine-tuned modes can run,
* output CSV files are generated.

The scores produced on this two-question sample benchmark are not the final project performance scores.

The final reported experimental results are provided in:

```text
reports/final_experiment_summary.md
```

In the project experiments, the Starlar fine-tuned Mistral QLoRA model improved controlled gold-context manual accuracy from `0.400` to `0.925`.

The sample benchmark is intentionally minimal and should not be interpreted as a representative evaluation.

## Notes

The `--retrieval_only` option can be used to verify that the custom document and benchmark pipeline works without loading the LLM.

Full base/fine-tuned RAG comparison requires:

1. Sufficient GPU memory.
2. Access to the base Mistral model.
3. The fine-tuned QLoRA adapter placed under `final_system/models/`.
4. `local_finetuned_adapter` configured correctly in `config.yaml`.
