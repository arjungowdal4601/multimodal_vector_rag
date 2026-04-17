# Multimodal Vector RAG

A multimodal PDF ingestion and retrieval pipeline for Retrieval-Augmented Generation (RAG).

This repository is a full end-to-end RAG system. It converts complex PDFs into retrieval-ready knowledge by preserving page structure, visual meaning, and cross-page continuity, stores them as embeddings in ChromaDB, and serves **grounded, verified answers** through a Streamlit chat UI.

The pipeline covers four stages:

1. **Document Processing** — convert a PDF into page-wise Markdown while preserving figures, tables, and formulas through generated descriptions.
2. **Page-Aware Chunking** — chunk Page N while using Page N+1 only as context to avoid breaking meaning at page boundaries.
3. **Vectorization** — embed the final chunk text and store it in ChromaDB.
4. **Retrieval & Grounded QA** — plan the user query, retrieve evidence per sub-question, draft grounded sub-answers with page screenshots, synthesize the final response, and verify every claim against retrieved evidence — all through a Streamlit chat UI.

---

## Why this project exists

Most PDF RAG systems fail early because they flatten a document into plain text too soon.

A real technical PDF contains:
- layout and page structure
- diagrams whose meaning lives in the image
- tables that lose meaning when converted badly
- formulas that matter semantically
- content that spills across page boundaries

This project is built on a simple idea:

> A PDF should be processed more like a human reads it, not like a string splitter reads it.

That means:
- page-wise processing
- visual asset preservation
- boundary-aware chunking
- cleaner embedding text for retrieval
- answers that stay grounded in the actual page they came from

---

## What is implemented

## 1. Document Processing (`doc_processor.py`)

The document processor works **page by page**.

### What it does
- loads a PDF
- enables OCR through Docling
- enables table structure extraction
- enables formula enrichment
- renders page screenshots
- extracts figure, table, and formula images
- generates dense descriptions for those assets using a vision-capable LLM
- reconstructs Markdown by replacing placeholders with:
  - image references
  - table references
  - formula references
  - retrieval-friendly descriptions

### Output
For a PDF like `sample.pdf`, the processor creates a folder like:

```text
sample_doc_assets/
├── page_images/
├── image_png_images/
├── table_images/
├── formula_images/
├── pages_md/
└── processed_doc.md
```

### Note on tables
The current tables are described in **dense prose** so their content remains searchable and useful during retrieval.

---

## 2. Page-Aware Chunking (`chunking.py`)

The chunker works page by page, but with limited next-page awareness.

### Core logic
For each pass:
- **Page N** is the target page
- **Page N+1** is optional context only
- the model chunks **only Page N**
- if a sentence, list, caption, or concept clearly continues across the page break, the chunker can pull in the **minimum necessary continuation** from Page N+1
- otherwise, the chunk stays grounded to Page N only

### Why this matters
A page break is not always a meaning break.

If chunking ignores that, the system creates broken chunks that are harder to retrieve correctly.

### Chunk schema
Each chunk stores:
- `content`
- `refreshed_content`
- `source`
- `metadata_addition`

Example shape:

```json
{
  "chunk_1": {
    "content": "raw chunk text",
    "refreshed_content": "rewritten self-contained chunk text for embeddings",
    "source": [1],
    "metadata_addition": {
      "pdf_name": "sample",
      "doc_assets_path": "sample_doc_assets"
    }
  }
}
```

### Output
Chunk JSON files are written page-wise into:

```text
sample_doc_assets/page_wise_json/
```

---

## 3. Vectorization (`vectorization.py`)

After chunking, the pipeline vectorizes the chunk text and stores it in ChromaDB.

### What it does
- loads all page-wise chunk JSON files
- prefers `refreshed_content`
- falls back to `content` if needed
- creates unique chunk IDs
- attaches metadata such as:
  - pages
  - pdf_name
  - doc_assets_path
  - page_json_file
  - chunk_id
- generates embeddings using OpenAI embeddings
- upserts everything into persistent ChromaDB

### Current storage config
- Chroma path: `vector_db/chroma_db`
- Collection name: `embeddings_db`
- Distance metric: cosine

---

## 4. Retrieval & Grounded QA (`app.py`)

After a PDF has been processed, chunked, and vectorized, `app.py` serves a **Streamlit chat UI** that takes a user question and returns a grounded, verified answer with page-level sources.

The query pipeline is:

```text
User query
 │
 ├── Memory context   (rolling conversation summary + recent turns)
 │
 ├── Query planning   (control LLM, structured output → QueryPlan)
 │     ├── standalone_query   (resolves follow-up references)
 │     ├── answer_instructions
 │     ├── sub_queries
 │     └── retrieval_tasks
 │
 ├── For each retrieval task:
 │     ├── embed the retrieval query
 │     ├── Chroma similarity search (top-k)
 │     ├── load page screenshots for retrieved chunks
 │     └── grounded sub-answer  (subanswer LLM, multimodal, structured output → GroundedSubAnswer)
 │           ├── supported  (bool)
 │           ├── answer_markdown
 │           └── source_pages
 │
 ├── Final synthesis   (final LLM, multimodal — combines sub-answers + deduped page images)
 │
 └── Verification      (control LLM — flags or rewrites unsupported claims → final_answer)
```

### What it does
- maintains a rolling conversation memory (summary + raw window of recent turns)
- resolves follow-up references in the user query ("what about those?") before retrieval
- decomposes multi-part queries into independent retrieval tasks
- retrieves the top `TOP_K` chunks per sub-question from ChromaDB
- drafts each sub-answer with the matching **page screenshots** included as multimodal input — the answer must cite the exact pages it used
- synthesizes the final answer from all grounded sub-answers
- runs a verification pass that checks every claim against the retrieved evidence and rewrites the answer if any claim is unsupported
- shows sources (page images) and a full debug expander in the UI — resolved query, retrieval plan, retrieved chunks, similarity scores

### Key data models (pydantic)
- `QueryPlan` — the planned decomposition of the user query
- `RetrievalTask` — a `(sub_question, retrieval_query)` pair
- `GroundedSubAnswer` — `{supported, answer_markdown, source_pages}`
- `VerificationResult` — `{had_unsupported_claims, verification_issues, corrected_answer_markdown}`
- `SplitTaskPlan` — used when a query is detected as multi-part and needs to be split further

### Configuration (hardcoded in `app.py`)

| Constant | Value | Purpose |
|---|---|---|
| `EMBED_MODEL` | `text-embedding-3-large` | Query + chunk embeddings |
| `CONTROL_MODEL` | `gpt-5.4-mini` | Query planning, memory summarization, verification |
| `SUBANSWER_MODEL` | `gpt-5.4` | Per-sub-question grounded answer (multimodal) |
| `FINAL_MODEL` | `gpt-5.4` (temp 0.1) | Final answer synthesis (multimodal) |
| `CHROMA_PATH` | `vector_db/chroma_db` | Persistent Chroma directory |
| `CHROMA_COLLECTION_NAME` | `embeddings_db` | Chroma collection |
| `PAGE_IMAGES_DIR` | `sample_doc_assets/page_images` | Where page screenshots are loaded from |
| `TOP_K` | `4` | Chunks retrieved per sub-question |
| `SUMMARY_BATCH_SIZE` | `3` | Turns between rolling-summary updates |
| `RAW_MEMORY_WINDOW` | `2` | Raw turns kept verbatim in memory context |

> **Note:** `PAGE_IMAGES_DIR` is currently hardcoded to `sample_doc_assets/page_images`. If you ingest multiple PDFs, you will want to route this off the chunk metadata (`doc_assets_path`) instead.

### Note on the vision model
Stages 1 and 2 (ingestion + chunking) still use the vision model configured via `OPENAI_VISION_MODEL` (default `gpt-5.2`). Stage 4 uses its own retrieval-side models (`gpt-5.4`, `gpt-5.4-mini`) hardcoded in `app.py`.

---

## Repository Structure

```text
multimodal_vector_rag/
├── app.py                       # Streamlit chat UI + retrieval / QA pipeline
├── doc_processor.py             # Stage 1 — PDF to page-wise Markdown
├── chunking.py                  # Stage 2 — page-aware semantic chunking
├── vectorization.py             # Stage 3 — embeddings + ChromaDB storage
├── main.ipynb                   # Ingestion driver notebook (runs stages 1-3)
├── requirements.txt
├── sample.pdf
├── sample_doc_assets/
│   ├── page_images/
│   ├── image_png_images/
│   ├── table_images/
│   ├── formula_images/
│   ├── pages_md/
│   ├── page_wise_json/
│   └── processed_doc.md
└── vector_db/
    └── chroma_db/
```

---

## End-to-End Flow

```text
PDF
 │
 ├── Stage 1: Document Processing
 │     ├── OCR
 │     ├── page screenshots
 │     ├── extract figures / tables / formulas
 │     ├── generate visual descriptions
 │     ├── export page-wise Markdown
 │     └── write processed assets
 │
 ├── Stage 2: Page-Aware Chunking
 │     ├── read page_N.md
 │     ├── read page_(N+1).md as context
 │     ├── optionally use screenshots
 │     ├── create semantic chunks
 │     ├── generate refreshed_content
 │     └── write page-wise JSON
 │
 ├── Stage 3: Vectorization
 │     ├── load chunk JSON files
 │     ├── embed final text
 │     ├── attach metadata
 │     └── store in ChromaDB
 │
 └── Stage 4: Retrieval & Grounded QA  (streamlit run app.py)
       ├── build memory context
       ├── plan the query → sub-queries + retrieval tasks
       ├── per sub-question:
       │     ├── embed → Chroma top-k
       │     ├── load page screenshots
       │     └── grounded sub-answer  (multimodal, structured)
       ├── synthesize final answer  (multimodal)
       ├── verify every claim against retrieved evidence
       └── render answer + source page images + debug view
```

---

## Setup

## 1. Clone the repository

```bash
git clone https://github.com/arjungowdal4601/multimodal_vector_rag.git
cd multimodal_vector_rag
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

### Windows
```bash
.venv\Scripts\activate
```

### macOS / Linux
```bash
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Add environment variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_VISION_MODEL=gpt-5.2
OPENAI_EMBED_MODEL=text-embedding-3-large
```

- `OPENAI_API_KEY` is required.
- `OPENAI_VISION_MODEL` is used by Stages 1 and 2 (defaults to `gpt-5.2`).
- `OPENAI_EMBED_MODEL` is used by Stage 3 (defaults to `text-embedding-3-large`).
- Stage 4 retrieval models (`gpt-5.4`, `gpt-5.4-mini`) are set inside `app.py` and can be changed there.

---

## Dependencies

Current pinned dependencies include:
- `docling`
- `docling-core`
- `langchain-core`
- `langchain-openai`
- `python-dotenv`
- `pypdf`
- `chromadb`
- `numpy`
- `streamlit`

---

## How to Run

Stages 1–3 can be run from `main.ipynb` or directly from Python. Stage 4 is the Streamlit app.

## Step 1 — Process the PDF

```python
from doc_processor import doc_processor_with_descriptions

doc_processor_with_descriptions("sample.pdf")
```

## Step 2 — Chunk the processed Markdown

```python
from chunking import chunk_markdown_with_llm

chunk_markdown_with_llm("sample.pdf")
```

## Step 3 — Vectorize and store in ChromaDB

```python
from vectorization import ingest_chunks_to_chroma

count = ingest_chunks_to_chroma("sample.pdf")
print(count)
```

## Step 4 — Launch the chat app

```bash
streamlit run app.py
```

This opens the chat UI in your browser. Ask a question about the ingested PDF and the app will plan the query, retrieve the relevant chunks, draft grounded sub-answers with page screenshots, synthesize the final answer, and run a verification pass before displaying it.

---

## Example Output

After running Stages 1–3 you will have:
- page screenshots
- extracted figure/table/formula images
- enriched page-wise Markdown
- stitched Markdown
- page-wise chunk JSON files
- embeddings stored in ChromaDB

After running Stage 4 (the Streamlit app) you will get:
- a chat answer grounded in the PDF
- the page images that were used as evidence
- a debug expander showing the resolved query, the planned sub-queries, the retrieved chunks, and their similarity scores

---

## Why `refreshed_content` exists

Embedding quality often improves when chunk text is more self-contained.

That is why each chunk stores both:
- `content`
- `refreshed_content`

The idea is to:
- preserve the original meaning
- make embedding text more standalone
- improve retrieval quality without dropping technical detail

---

## Roadmap

### Done
- ingestion pipeline (Stages 1–3)
- retrieval + grounded QA layer (Stage 4)
- Streamlit chat UI with source page display and debug view
- rolling conversation memory
- grounded sub-answering with page screenshots as multimodal context
- verification pass over every final answer

### Near-term
- **reduce end-to-end query latency** — parallel fan-out over sub-questions, batched embeddings, conditional verification
- move to a LangGraph-based orchestration with subgraphs (planning subgraph + retrieval/QA subgraph with `Send` fan-out), and stream node-level updates to the UI
- route `PAGE_IMAGES_DIR` off each chunk's `doc_assets_path` metadata so multiple PDFs work cleanly
- add caching for repeated queries, embeddings, and page-image base64 encoding
- add evaluation metrics and compare page-aware chunking against naive baselines

### Long-term
- support more document types
- improve table-to-structured-fact conversion
- make multimodal retrieval more explainable and robust

---

## Final Note

This repository is based on one simple belief:

> If you want better answers from RAG, start by building better chunks.

That is what this project is trying to do — from the first page of ingestion all the way through to the verified answer on screen.
