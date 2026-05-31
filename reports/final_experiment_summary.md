# Final Experiment Summary

This document summarizes the final experimental results of the Turkish Legal RAG project. The project includes retrieval improvements, reranker fine-tuning, LLM fine-tuning, ablation studies, and a final runnable custom RAG benchmark workspace.

## 1. Best End-to-End RAG Result

The best completed end-to-end RAG pipeline was obtained on the original benchmark using the following configuration:

- Source-aware retrieval
- Article-aware retrieval
- Turkish BGE reranker
- Improved legal prompt
- Base Mistral generator

This setting represents the best complete RAG pipeline tested on the original benchmark dataset.

| Setting | Manual Accuracy |
|---|---:|
| Original benchmark | 0.421 |
| Coverage-clean benchmark | 0.46875 |

The coverage-clean score excludes known problematic samples caused by missing, invalid, or mismatched corpus coverage. Therefore, the original benchmark score represents the full evaluation result, while the coverage-clean score shows the performance after removing samples where the expected answer was not properly supported by the available corpus.

## 2. Retrieval and Reranking Experiments

Several retrieval and reranking configurations were tested during the project. The main goal was to improve the quality of retrieved legal contexts before answer generation.

The experiments included:

- Baseline retrieval
- Hybrid retrieval
- Turkish BGE reranking
- Source-aware retrieval
- Article-aware retrieval
- Improved legal prompting
- Fine-tuned Turkish BGE reranker

The reranker fine-tuning experiment improved ranking quality on the validation setup. The fine-tuned Turkish BGE reranker achieved a strong validation AUC, but the final end-to-end RAG score was still limited by corpus coverage, benchmark-question mismatch, and answer-generation quality.

The final selected end-to-end RAG pipeline used source-aware and article-aware retrieval together with Turkish BGE reranking and the improved legal prompt.

## 3. LLM Fine-Tuning Result

LLM fine-tuning was evaluated separately under a controlled gold-context setting using the Starlar LLM SFT dataset.

In this setting, the correct source context was already provided to the model. Therefore, the experiment focused specifically on source-grounded answer generation rather than retrieval quality.

The same base LLM family was used for comparison:

- Base model: Mistral-7B-Instruct-v0.2
- Fine-tuned model: Mistral-7B-Instruct-v0.2 + Starlar QLoRA adapter

The controlled evaluation results were:

| Model | Manual Accuracy | Mean Token F1 | Mean Text Similarity | Source Citation Rate |
|---|---:|---:|---:|---:|
| Base Mistral | 0.400 | 0.1848 | 0.1554 | 0.25 |
| Starlar Fine-tuned Mistral QLoRA | 0.925 | 0.9657 | 0.9695 | 0.85 |

These results show that LLM fine-tuning substantially improved source-grounded Turkish legal answer generation when the correct context was provided.

## 4. Integration Attempt on the Original RAG Benchmark

After obtaining strong controlled fine-tuning results, the Starlar fine-tuned LLM was also tested on the fixed retrieved contexts from the previous best RAG pipeline.

The purpose of this experiment was to check whether the fine-tuned LLM could directly replace the base generator in the original end-to-end RAG benchmark.

However, this direct integration did not improve the original benchmark result.

The main reasons were:

- The Starlar fine-tuning dataset and the original RAG benchmark had different answer styles.
- The Starlar dataset encouraged source/citation-style answers.
- The original benchmark expected shorter and more direct answers.
- Some original benchmark samples had retrieval or context coverage problems.
- Some expected answers were not fully supported by the retrieved contexts.

Because of these differences, the Starlar fine-tuned LLM performed strongly in the controlled gold-context evaluation, but it was not selected as the final generator for the original RAG benchmark.

## 5. Target-Style Alignment Ablation

A target-style alignment fine-tuning experiment was also attempted.

The goal was to adapt the Starlar fine-tuned model to the shorter, direct-answer style of the original RAG benchmark. For this purpose, a smaller target-style SFT dataset was prepared from high-quality context-question-answer records.

This dataset used the following answer style:

- Short Turkish answers
- Direct answer format
- No source/citation lines
- No chunk IDs
- No copied context blocks

The model was trained for a short alignment run starting from the Starlar fine-tuned adapter.

Although validation loss decreased during this alignment run, qualitative sanity checks showed that the resulting adapter produced generic, repetitive, or legally inaccurate answers in several cases. Therefore, the target-aligned adapter was not selected as the final RAG generator.

This experiment was kept as an ablation study and not used as the final system configuration.

## 6. Final System Decision

The final project reports two complementary findings:

1. Retrieval and prompt engineering produced the best end-to-end RAG result on the original benchmark.
2. LLM fine-tuning clearly improved controlled source-grounded answer generation on the Starlar dataset.

The final selected end-to-end RAG result remains:

| Component | Selected Method |
|---|---|
| Retrieval | Source-aware + article-aware retrieval |
| Reranking | Turkish BGE reranker |
| Prompting | Improved legal prompt |
| Generator for original benchmark | Base Mistral |
| Fine-tuned LLM experiment | Starlar fine-tuned Mistral QLoRA |

The Starlar fine-tuned LLM is reported as a successful controlled fine-tuning result, but it is not used as the final generator for the original RAG benchmark because direct integration did not improve the end-to-end score.

## 7. Custom Evaluation Support

## 7. Custom Evaluation Support

A runnable custom RAG benchmark workspace is provided under:

```text
final_system/
```

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

The custom runner supports the following modes:

- `base`: runs the RAG system with the base Mistral model.
- `finetuned`: runs the RAG system with the same base Mistral model plus a fine-tuned QLoRA adapter.
- `both`: runs both systems on the same benchmark and saves comparison outputs.

Example retrieval-only test:

```bash
cd final_system
python run_custom_rag_benchmark.py --mode base --retrieval_only
```

This mode verifies that custom documents can be loaded, chunked, retrieved, and evaluated without loading the LLM.

If a fine-tuned adapter is configured in `config.yaml`, the instructor can also run:

```bash
python run_custom_rag_benchmark.py --mode both
```

The generated outputs are saved under:

```text
final_system/data/outputs/
```

## 8. Final Notes

The project includes both experimental notebooks and a final runnable system.

The notebooks document the research and experimentation process, including retrieval experiments, reranker fine-tuning, LLM fine-tuning, controlled evaluation, integration attempts, and ablation studies.

The `final_system/` folder provides a clean interface for running the RAG system on custom documents and custom benchmark questions.
