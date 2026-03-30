# Multimodal Vector RAG

A multimodal PDF ingestion pipeline for Retrieval-Augmented Generation (RAG).

This repository focuses on the **document preparation layer** of a RAG system. It is built to convert complex PDFs into retrieval-ready knowledge by preserving page structure, visual meaning, and cross-page continuity before vector storage.

The current pipeline covers three core stages:

1. **Document Processing** — convert a PDF into page-wise Markdown while preserving figures, tables, and formulas through generated descriptions.
2. **Page-Aware Chunking** — chunk Page N while using Page N+1 only as context to avoid breaking meaning at page boundaries.
3. **Vectorization** — embed the final chunk text and store it in ChromaDB.

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

---

## Repository Structure

```text
multimodal_vector_rag/
├── doc_processor.py
├── chunking.py
├── vectorization.py
├── main.ipynb
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
 └── Stage 3: Vectorization
       ├── load chunk JSON files
       ├── embed final text
       ├── attach metadata
       └── store in ChromaDB
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

You can run the pipeline from `main.ipynb` or directly from Python.

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

---

## Example Output

After running the pipeline you will have:
- page screenshots
- extracted figure/table/formula images
- enriched page-wise Markdown
- stitched Markdown
- page-wise chunk JSON files
- embeddings stored in ChromaDB

---

## UI and Next Phase

The current repository is centered on the ingestion and vectorization pipeline.

A future app layer can sit on top of this backend and use:
- processed Markdown
- page images
- page-wise chunk JSON files
- ChromaDB metadata and embeddings

That next layer can handle:
- query-time retrieval
- grounded answer generation
- source page display
- user-facing interaction through a Streamlit app

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

### Near-term
- add retrieval layer
- add a real UI
- show grounded answers with page evidence
- support question-answer flow over processed PDFs

### Next serious step
- compare page-aware chunking against naive chunking baselines
- add evaluation metrics
- reduce latency and token cost
- add caching and incremental processing

### Long-term
- support more document types
- improve table-to-structured-fact conversion
- make multimodal retrieval more explainable and robust

---

## Final Note

This repository is based on one simple belief:

> If you want better answers from RAG, start by building better chunks.

That is what this project is trying to do.
