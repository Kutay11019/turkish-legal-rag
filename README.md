## How to Run and Use the Final System

This repository contains the final runnable Turkish Legal RAG system. The recommended way to test the system is through the **Streamlit demo interface**.

The Streamlit interface allows the instructor to:

* upload custom legal files,
* ask a single legal question,
* generate one final answer,
* inspect retrieved legal sources,
* optionally run a batch benchmark file.

Full answer generation uses **Mistral-7B-Instruct**, which requires GPU memory. Therefore, the recommended setup is:

```text
Local computer:
Streamlit user interface

Google Colab:
GPU backend for retrieval, reranking, and Mistral generation
```

The instructor interacts only with the Streamlit interface. Google Colab is used only as the GPU backend.

---

# Streamlit Demo Interface

## 1. Open the Repository Locally

Clone this repository to your computer and open the project folder in VS Code.

The local Streamlit interface should be run from the repository root.

Important files for the demo:

| File                              | Purpose                                     |
| --------------------------------- | ------------------------------------------- |
| `streamlit_demo.py`               | Local Streamlit user interface              |
| `colab_rag_backend.py`            | Backend code used by the Colab GPU notebook |
| `requirements_local_frontend.txt` | Local Streamlit interface dependencies      |
| `requirements_colab_backend.txt`  | Colab backend dependencies                  |

---

## 2. Install Local Interface Dependencies

Open the VS Code terminal in the repository root.

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it.

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the local interface dependencies:

```bash
pip install -r requirements_local_frontend.txt
```

If needed, the minimum local dependencies can also be installed manually:

```bash
pip install streamlit requests pandas
```

The local interface does not load Mistral locally. It only displays the user interface and sends requests to the Colab GPU backend.

---

## 3. Start the Colab Backend

The full model generation should be run on Google Colab because the LLM requires GPU memory.

Open the Colab backend notebook:

[Open Final System Colab Backend](https://colab.research.google.com/drive/1_qViXMgkFK_2OW21GhjJMgOFlbWoGU0g#scrollTo=IGxNg8qv1Sik)

Before running the notebook, select a GPU runtime:

```text
Runtime > Change runtime type > Hardware accelerator > GPU
```

An L4 or A100 GPU is recommended when available.

Then run the notebook cells from top to bottom.

The notebook will:

* access or clone the project files,
* install the required backend libraries,
* load the required retrieval, reranking, and generation components,
* start the backend service,
* expose the backend through a public URL.

After the backend starts, copy the generated public backend URL from the Colab output.

Example URL formats:

```text
https://xxxxx.trycloudflare.com
```

or:

```text
https://xxxxx.ngrok-free.app
```

The exact URL changes each time the Colab backend is restarted.

Important notes:

* Keep the Colab notebook running while using the Streamlit interface.
* If the backend is restarted, a new public URL may be generated.
* Copy only the main public URL.
* Do not add `/health`, `/answer`, or any other endpoint suffix when pasting it into the Streamlit interface.

---

## 4. Start the Local Streamlit Interface

In VS Code, run the Streamlit interface from the repository root:

```bash
streamlit run streamlit_demo.py
```

Streamlit will open a local browser page such as:

```text
http://localhost:8501
```

---

## 5. Connect the Interface to the Colab Backend

In the Streamlit interface:

1. Find the **Colab GPU backend URL** field.
2. Paste the public backend URL copied from Colab.
3. Click **Check backend**.
4. If the connection is successful, the interface will show backend and GPU status.

After this step, the local interface is connected to the GPU backend.

---

# Using the Streamlit Interface

## 1. Upload Legal Files

The Streamlit interface supports legal source files in the following formats:

```text
.txt
.pdf
.docx
.csv
.json
.jsonl
```

Multiple files can be uploaded at the same time. They can be the same file type or different file types.

Examples:

```text
constitution.pdf
tck_articles.docx
legal_articles.csv
regulation.json
custom_rules.jsonl
```

The uploaded files are used as the retrieval corpus. The system reads the files, extracts text, creates chunks or structured records, retrieves relevant legal contexts, reranks them, and generates an answer based on the selected context.

For structured files such as CSV, JSON, and JSONL, each row or record is treated as a separate retrievable unit when possible. This helps prevent unrelated records from being mixed into the same chunk.

Example legal CSV file:

```csv
madde_no,baslik,metin
1,Başvuru Süresi,"İdari başvuru süresi, kararın ilgili kişiye tebliğinden itibaren 30 gündür."
2,İtiraz Süresi,"Başvurunun reddedilmesi halinde kişi, ret kararının tebliğinden itibaren 15 gün içinde ilgili kuruma yazılı itirazda bulunabilir."
3,Belge Saklama,"Kurumlar, başvuruya ilişkin belgeleri en az 5 yıl süreyle saklamak zorundadır."
```

---

## 2. Ask a Single Question

After uploading legal files:

1. Type a question in the question box.
2. Click **Generate Answer**.
3. The system returns one final generated answer.
4. Open **Retrieved Legal Sources** if you want to inspect the retrieved evidence.

Example question:

```text
Başvuru reddedilirse kaç gün içinde itiraz edilebilir?
```

Expected answer from the example CSV:

```text
Başvurunun reddedilmesi halinde kişi, ret kararının tebliğinden itibaren 15 gün içinde ilgili kuruma yazılı itirazda bulunabilir.
```

If the uploaded documents do not contain the answer, the system should respond that the answer cannot be found in the provided context.

---

## 3. Retrieved Legal Sources

After generation, the interface can show the retrieved legal sources.

This section helps verify:

* which uploaded file was used,
* which chunk or structured record was retrieved,
* whether the retrieved context supports the generated answer,
* whether the system answered from the provided legal source instead of unsupported model knowledge.

This is important because the project is a legal RAG system. The generated answer should be grounded in retrieved legal text.

---

# Optional Batch Benchmark Test

The Streamlit interface also supports an optional benchmark mode.

A benchmark file is different from a legal source file.

```text
Legal file:
The document or corpus used for retrieval and answering.

Benchmark file:
A list of questions used to test the system in batch mode.
```

Supported benchmark formats:

```text
.csv
.json
.jsonl
```

Each benchmark sample should contain at least a question field. It may also contain an expected answer field.

Supported field names include:

```text
question
expected_answer
```

or Turkish equivalents such as:

```text
soru
cevap
```

Example benchmark CSV:

```csv
question,expected_answer
"İdari başvuru süresi kaç gündür?","İdari başvuru süresi 30 gündür."
"Başvuru reddedilirse kaç gün içinde itiraz edilebilir?","Ret kararının tebliğinden itibaren 15 gün içinde itiraz edilebilir."
"Kurumlar belgeleri kaç yıl saklar?","Kurumlar belgeleri en az 5 yıl saklamak zorundadır."
```

To run a benchmark in the interface:

1. Upload at least one legal source file in the main legal file upload section.
2. Open **Optional: Batch Benchmark Test**.
3. Upload a CSV, JSON, or JSONL benchmark file.
4. Select the maximum number of questions to run.
5. Click **Run Benchmark**.
6. View the generated answers in the table.
7. Download the results as a CSV file.

Important:

```text
A benchmark file alone is not enough.
The system also needs at least one legal source file to retrieve evidence from.
```

---

# Models Used in the Streamlit Demo

The current Streamlit demo uses the following main components:

| Component       | Model / Method                                                |
| --------------- | ------------------------------------------------------------- |
| Generator       | `mistralai/Mistral-7B-Instruct-v0.2`                          |
| Embedding model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Reranker        | Turkish BGE reranker (`seroe/bge-reranker-v2-m3-turkish-triplet`)                    |
| Retrieval       | Hybrid retrieval with source-aware and article-aware logic    |
| Prompting       | Improved legal grounding prompt                               |

The demo uses the **base Mistral generator** because the best original end-to-end RAG result was obtained with base Mistral combined with improved retrieval, Turkish BGE reranking, and improved legal prompting.

The fine-tuned Starlar QLoRA adapter is not used as the default Streamlit demo generator because it improved controlled gold-context generation but did not improve the original end-to-end RAG benchmark when directly integrated.

---

# Optional Command Line Runner

The original command line runner is still available under:

```text
final_system/
```

For a quick retrieval-only smoke test without loading the LLM:

```bash
cd final_system
python run_custom_rag_benchmark.py --mode base --retrieval_only
```

This checks whether documents can be loaded, chunked, retrieved, and written to output files. This mode can run on CPU.

For full base or fine-tuned generation from the command line, use a CUDA-enabled GPU environment or the provided Colab workflow.

Detailed fine-tuned adapter setup is not required for the Streamlit demo. The fine-tuned adapter is stored outside GitHub due to file size and is used only for optional advanced experiments.

---

# Final Results and Metrics

The project reports two final result groups:

1. **Best End-to-End RAG Result**
2. **Controlled LLM Fine-Tuning Result**

These two results measure different evaluation settings and should not be interpreted as the same experiment.

The end-to-end RAG result evaluates the complete pipeline:

```text
question → retrieval → reranking → context selection → answer generation
```

The controlled fine-tuning result evaluates generation separately when the correct legal context is already provided:

```text
question + correct context → answer generation
```

Therefore, the fine-tuned LLM score is higher because the correct legal context was already supplied to the model.

---

## 1. Best End-to-End RAG Result

The best completed original benchmark pipeline used:

* source-aware retrieval,
* article-aware retrieval,
* Turkish BGE reranker,
* improved legal prompt,
* base Mistral generator.

| Evaluation Setting       | Manual Accuracy | Success Percentage |
| ------------------------ | --------------: | -----------------: |
| Original benchmark       |           0.421 |             42.10% |
| Coverage-clean benchmark |         0.46875 |            46.875% |

The coverage-clean benchmark excludes known problematic samples where the correct answer was missing from the corpus or where the question/context match was inconsistent.

This result represents the best complete RAG pipeline performance on the original project benchmark.

---

## 2. Controlled LLM Fine-Tuning Result

LLM fine-tuning was evaluated separately using the Starlar LLM SFT dataset.

In this setup, the correct context was already given to the model. Therefore, this experiment measures whether the model can generate a correct, source-grounded Turkish legal answer when the relevant legal context is available.

| Model                            | Manual Accuracy | Success Percentage | Mean Token F1 | Mean Text Similarity | Source Citation Rate |
| -------------------------------- | --------------: | -----------------: | ------------: | -------------------: | -------------------: |
| Base Mistral                     |           0.400 |             40.00% |        0.1848 |               0.1554 |                 0.25 |
| Starlar Fine-tuned Mistral QLoRA |           0.925 |             92.50% |        0.9657 |               0.9695 |                 0.85 |

This shows that fine-tuning substantially improved source-grounded Turkish legal answer generation when the correct context was provided.

However, the fine-tuned adapter did not improve the original end-to-end RAG benchmark when directly integrated into the older best RAG context pipeline. The likely reasons were dataset distribution differences and answer-format mismatch.

---

# Metric Explanations

| Metric               | Meaning                                                                                          |
| -------------------- | ------------------------------------------------------------------------------------------------ |
| Manual Accuracy      | Human evaluation score showing whether the generated answer is legally and semantically correct. |
| Mean Token F1        | Token-level overlap between the generated answer and the expected answer.                        |
| Mean Text Similarity | Semantic/textual similarity between the generated answer and the expected answer.                |
| Source Citation Rate | How often the answer includes or correctly refers to the provided source/context.                |

---

# Final Interpretation

The project has two main conclusions:

1. The best end-to-end RAG pipeline achieved the strongest full-system performance using retrieval improvements, reranking, prompt engineering, and the base Mistral generator.
2. Fine-tuning the LLM significantly improved answer generation quality in a controlled setting where the correct context was already supplied.

The fine-tuned model was also tested inside the older best RAG pipeline, but it did not improve the final end-to-end benchmark result. The likely reasons were:

* dataset distribution differences,
* answer-format mismatch,
* retrieval/context coverage limitations,
* some benchmark questions not being fully supported by the available corpus.

Therefore, the final project reports both results separately:

* **Best end-to-end RAG result:** complete pipeline performance.
* **Fine-tuned LLM result:** controlled generation performance with gold context.
