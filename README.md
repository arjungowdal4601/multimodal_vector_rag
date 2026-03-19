# Multimodal RAG from First Principles

> Most RAG pipelines fail before retrieval even starts.
>
> They fail when they flatten a document that was never flat.
>
> This repository exists because PDFs are not raw text files. They are structured artifacts made of pages, reading order, tables, figures, formulas, captions, and broken context across page boundaries. If you chunk them like plain text, you destroy meaning before the vector database ever sees it.
>
> This project is an opinionated attempt to fix that.

---

## What this repository is

This is a **multimodal document ingestion pipeline for RAG**.

It takes a PDF and turns it into retrieval-ready knowledge through three completed stages:

1. **Document processing** — convert each page into markdown and enrich visual elements with retrieval-friendly descriptions.
2. **Page-aware chunking** — chunk page by page, while looking one page ahead to repair context broken by page boundaries.
3. **Vectorization** — embed the final chunks and store them in ChromaDB.

The **UI and retrieval layer are intentionally not the focus of this stage yet**. This repository is about getting the knowledge representation right before building the chatbot on top.

---

## Why this project exists

### The real problem

Most "PDF RAG" demos are dishonest.

They make retrieval look easy because they quietly assume the source document is clean text. Real PDFs are not.

A real technical PDF contains:

- text in layout order, not logical order
- figures whose meaning lives in the image, not the surrounding sentence
- tables whose value disappears when converted into broken markdown
- formulas that matter semantically but get lost during parsing
- paragraphs, bullets, and captions that continue across page boundaries

If you ignore that structure, then your retrieval system is already compromised.

### First-principles view

RAG is not magic. It is a pipeline with four basic responsibilities:

1. **Observe the source correctly**
2. **Represent the source without destroying meaning**
3. **Split the source into retrievable units**
4. **Store those units so a retriever can find the right one later**

Most people obsess over step 4.

This project obsesses over steps 1 to 3.

That is the difference.

---

## Core idea

The central claim of this repository is simple:

> **A PDF should be processed the way a careful human reads it, not the way a string splitter sees it.**

A human does not consume a technical document as one giant text blob.

A human naturally thinks in:

- pages
- sections
- local continuity
- visual references
- manageable units of meaning

That is why this pipeline is built around:

- **page-wise document processing**
- **page-aware chunking**
- **cross-page boundary repair**
- **refreshed chunk text for embeddings**

---

## What is completed right now

### Stage 1 — Document processing

The PDF is processed **one page at a time**.

For each page, the pipeline:

- renders and saves the page image
- extracts images, tables, and formulas
- generates retrieval-friendly descriptions for those visual assets using a vision model
- exports the page into markdown
- replaces placeholders with the actual asset references and generated descriptions
- saves both:
  - a stitched markdown document for the whole PDF
  - an individual markdown file for each page

This design is directly reflected in `doc_processor.py`, which creates folders such as `page_images`, `image_png_images`, `table_images`, `formula_images`, `pages_md`, and a final stitched markdown file under `<pdf_name>_doc_assets`. It enables OCR, table structure extraction, formula enrichment, page images, picture images, and table images through Docling’s PDF pipeline options. fileciteturn5file0

### Stage 2 — Chunking

The chunker reads **one page as the target page** and **the next page as context only**.

That means:

- Page N is the page being chunked.
- Page N+1 is not independently chunked in that pass.
- Page N+1 is only used to check whether meaning spills across the page boundary.

If the last idea on Page N clearly continues into Page N+1, the chunk is allowed to pull in the **minimum necessary continuation text** and store both page numbers as sources.

If not, the chunk remains grounded in the target page only.

This is the key design choice in `chunking.py`: the target page and optional next page are both sent to the vision-capable model, which is instructed to chunk only the target page, repair cross-page flow only when needed, and return strict JSON containing `content`, `refreshed_content`, and `source` page lists. The page-wise JSON files are written into `page_wise_json` under the same asset root. fileciteturn5file2

### Stage 3 — Vectorization

After chunking, the pipeline loads every page-wise JSON file, prefers `refreshed_content` over raw `content`, generates embeddings with OpenAI, and upserts the results into a persistent Chroma collection.

The current implementation stores metadata such as page numbers, PDF name, doc-assets path, page JSON file, and chunk ID, then writes the embeddings into `vector_db/chroma_db` using a shared collection named `embeddings_db`. fileciteturn5file1

---

## What is *not* done yet

This repository is **not pretending to be finished**.

The following are still pending or intentionally deferred:

- Streamlit UI
- retrieval chain / answer generation layer
- evaluation harness for retrieval quality
- benchmark comparison against naive chunking baselines
- caching and cost controls for repeated LLM-heavy processing
- production hardening

If you are looking for a polished chatbot demo, this repo is not there yet.

If you care about how a document should be converted into something a retriever can trust, this repo is exactly about that.

---

## Architecture

```text
PDF
  │
  ├── Stage 1: Document Processing
  │     ├── render page images
  │     ├── extract pictures / tables / formulas
  │     ├── describe visuals with vision model
  │     ├── export page-wise markdown
  │     └── write <pdf_name>_doc_assets/
  │
  ├── Stage 2: Page-Aware Chunking
  │     ├── read page_N.md
  │     ├── read page_(N+1).md as context
  │     ├── detect cross-page overflow
  │     ├── create semantic chunks
  │     ├── refresh chunk text for embeddings
  │     └── write page-wise JSON
  │
  └── Stage 3: Vectorization
        ├── load page-wise JSON chunks
        ├── embed refreshed text
        ├── attach metadata
        └── upsert into ChromaDB
```

---

## Repository philosophy

### 1. Do not flatten structure too early

A PDF has layout, hierarchy, and visual context.

The moment you flatten all of that into plain text too early, you lose information you will never get back.

### 2. Chunk by meaning, not by character count

Character-based chunking is simple, fast, and frequently wrong.

It is useful as a baseline, not as a default for serious technical documents.

### 3. Page is a useful cognitive unit

A page is not a perfect semantic boundary.

But it is a **real boundary** that human readers understand, authors design around, and PDFs physically encode. That makes it a better starting point than arbitrary token windows for many technical documents.

### 4. Boundary repair matters

A page break is not a meaning break.

If a sentence, caption, bullet list, or explanation continues onto the next page, your chunking logic must notice that. Otherwise you create half-truth chunks.

### 5. Embedding text should stand on its own

That is why this project stores both:

- `content` — the original chunk text
- `refreshed_content` — a rewritten, self-contained version meant to preserve meaning more clearly for embeddings

This is not decoration. It is an attempt to make the embedding input more semantically stable.

---

## Output folder structure

Running the pipeline on `sample.pdf` will create something like:

```text
sample_doc_assets/
├── page_images/
│   ├── page-1.png
│   ├── page-2.png
│   └── ...
├── image_png_images/
│   ├── picture-1.png
│   └── ...
├── table_images/
│   ├── table-1.png
│   └── ...
├── formula_images/
│   ├── formula-1.png
│   └── ...
├── pages_md/
│   ├── page_0001.md
│   ├── page_0002.md
│   └── ...
├── page_wise_json/
│   ├── page_0001.json
│   ├── page_0002.json
│   └── ...
└── processed_doc.md
```

This folder structure is a direct consequence of the current processor and chunker implementations. fileciteturn5file0 fileciteturn5file2

---

## Example chunk JSON shape

Each page-wise JSON file stores chunks in a structure like this:

```json
{
  "chunk_1": {
    "content": "raw chunk text",
    "refreshed_content": "self-contained rewritten chunk for embeddings",
    "source": [1],
    "metadata_addition": {
      "pdf_name": "sample",
      "doc_assets_path": "sample_doc_assets"
    }
  },
  "chunk_2": {
    "content": "chunk with boundary continuation",
    "refreshed_content": "refreshed meaning-preserving text",
    "source": [1, 2],
    "metadata_addition": {
      "pdf_name": "sample",
      "doc_assets_path": "sample_doc_assets"
    }
  }
}
```

The exact fields are written by `_save_page_json()` in `chunking.py`, and the vectorization step later consumes these fields when creating metadata for Chroma. fileciteturn5file2 fileciteturn5file1

---

## How the three main scripts work

## 1) `doc_processor.py`

### Job
Turn a PDF into page-wise markdown plus retrieval-friendly visual descriptions.

### What it does internally

- loads environment variables
- builds a Docling converter with OCR, table structure extraction, formula enrichment, and page-image generation
- processes the PDF one page at a time
- saves page images
- extracts visual assets by type
- calls the vision model to describe:
  - pictures
  - tables
  - formulas
- injects those descriptions back into markdown
- saves per-page markdown and a stitched markdown file

### Why it matters

Retrieval quality depends on what the chunker sees.

If your chunker only sees weak OCR text and loses the figure or table meaning, retrieval will miss answers that obviously exist in the source.

---

## 2) `chunking.py`

### Job
Turn page-wise markdown into semantic, page-aware chunks.

### What it does internally

- reads markdown from `pages_md/`
- reads rendered page images from `page_images/`
- takes one target page and optionally the next page
- sends both text and page screenshots to a vision-capable LLM
- instructs the model to chunk only the target page
- allows minimal continuation from the next page when cross-page flow is real
- forces strict JSON output
- stores both original and refreshed chunk text
- writes page-wise chunk JSON files

### Why it matters

This is where the project stops behaving like a string-splitting demo and starts behaving like a document-understanding pipeline.

---

## 3) `vectorization.py`

### Job
Embed chunk text and persist it into a vector database.

### What it does internally

- locates the chunk JSON files for a given PDF
- loads all chunks page by page
- prefers `refreshed_content` as the embedding text
- builds stable unique IDs across PDFs and pages
- converts `source` pages into scalar metadata that Chroma can store
- generates embeddings with OpenAI
- upserts the final records into persistent ChromaDB

### Why it matters

Without stable metadata and consistent IDs, retrieval becomes harder to trace and debug. This step is not just about storing vectors. It is about storing **recoverable evidence**.

---

## Tech stack

This repository currently depends on the following major components:

- **Docling** for PDF parsing, OCR, structure extraction, and markdown export. Docling’s README highlights advanced PDF understanding, table structure, formulas, multiple export formats, and local execution capabilities. citeturn907798search1turn907798search6
- **OpenAI vision-capable chat model** for describing figures, tables, formulas, and performing page-aware chunking. This is visible in both `doc_processor.py` and `chunking.py`, where `ChatOpenAI` is used with `OPENAI_VISION_MODEL`. fileciteturn5file0 fileciteturn5file2
- **OpenAI embeddings** for converting the final chunk text into vectors, using `OpenAIEmbeddings` with a configurable embedding model. fileciteturn5file1
- **ChromaDB** as the persistent vector store. Chroma’s README positions it as an open-source embedding database with a simple API for collections, adds, and queries. citeturn319240search4
- **Streamlit** is the planned UI layer. Streamlit’s README emphasizes fast interactive prototyping from simple Python scripts, which is why it is a reasonable choice for the upcoming interface. citeturn319240search1

---

## Setup

### 1) Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2) Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3) Install dependencies

If you maintain a `requirements.txt`:

```bash
pip install -r requirements.txt
```

If not, your current code clearly requires at least libraries corresponding to Docling, LangChain OpenAI integration, ChromaDB, `python-dotenv`, and `pypdf`, because those modules are imported directly in the project files. fileciteturn5file0 fileciteturn5file1 fileciteturn5file2

### 4) Create a `.env` file

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_VISION_MODEL=gpt-5.2
OPENAI_EMBED_MODEL=text-embedding-3-large
```

These environment variables are referenced in the current scripts. `OPENAI_VISION_MODEL` and `OPENAI_EMBED_MODEL` both have defaults in code, but `OPENAI_API_KEY` must be set for the OpenAI calls to work. fileciteturn5file0 fileciteturn5file1 fileciteturn5file2

---

## How to run the current pipeline

### Step 1 — Process the PDF

```python
from doc_processor import doc_processor_with_descriptions

doc_processor_with_descriptions("sample.pdf")
```

### Step 2 — Chunk the processed markdown

```python
from chunking import chunk_markdown_with_llm

chunk_markdown_with_llm("sample.pdf")
```

### Step 3 — Vectorize the chunks

```python
from vectorization import ingest_chunks_to_chroma

ingest_chunks_to_chroma("sample.pdf")
```

After these three steps, you will have:

- page-wise markdown
- page-wise chunk JSON
- persistent embeddings stored in ChromaDB

---

## Why not just use recursive chunking?

Because simplicity is not the same as correctness.

Recursive or character-based chunking is often fine for:

- clean prose
- blog posts
- web text
- documents where layout barely matters

It is often weaker for:

- technical PDFs
- page-heavy reports
- documents with tables, images, and formulas
- documents where captions, bullets, and paragraphs overflow across pages

This repository does **not** claim that page-aware chunking is universally superior.

It claims something narrower and more credible:

> For structured technical PDFs with meaningful page boundaries and cross-page overflow, page-aware chunking is a better design starting point than blind text splitting.

That is a stronger claim because it is actually defensible.

---

## Limitations

This project is useful, but it is not cheap, fast, or fully production-ready yet.

### 1. LLM-heavy pipeline
The processor and chunker rely on OpenAI calls for visual descriptions and chunk creation. That means cost, latency, and rate-limit sensitivity.

### 2. No evaluation harness yet
The repo does not yet ship with retrieval benchmarks, golden questions, recall metrics, or chunking comparisons.

### 3. No caching layer yet
If you rerun the pipeline carelessly, you can pay repeatedly for work you already did.

### 4. Current vectorization is single-path
Right now the vectorization flow is tightly coupled to OpenAI embeddings and Chroma storage.

### 5. UI and retrieval are not finished
This is currently an ingestion-first repository, not a full user-facing application.

### 6. Markdown heuristics are still heuristics
Table and formula placeholder replacement is structured, but still based on practical pattern matching and model behavior, not on perfect symbolic understanding.

---

## Roadmap

### Near-term
- Build Streamlit UI
- Add retrieval pipeline
- Show source pages in final answers
- Add question-answer demo over processed PDFs

### Next serious step
- Compare against baseline chunkers
- Measure chunk quality and retrieval hit rate
- Add caching and incremental processing
- Support alternative embedding models
- Support local-first or hybrid execution

### Long-term
- move from “works on my PDF” to “reliable on a class of technical documents”
- add evaluation data and benchmarks
- support more document types
- make retrieval explainable with grounded citations to page assets and chunks

---

## Who this is for

This repository is for people who are tired of shallow RAG demos.

It is for builders who care about:

- document structure
- multimodal ingestion
- chunk quality
- retrieval traceability
- getting the foundation right before building UI hype on top of it

It is not for people who want a two-minute "chat with your PDF" clone and do not care what the system destroys on the way.

---

## Recommended repository structure and documentation notes

GitHub’s own documentation says a repository README should explain what the project does, why it is useful, how to get started, where to get help, and who maintains it. It also notes that GitHub surfaces README files automatically when placed in `.github`, the repository root, or `docs/`. citeturn907798search0

That is why this README is structured around:

- problem
- architecture
- setup
- usage
- limitations
- roadmap

That is also how many strong open-source READMEs are written: they lead with purpose, then give installation or quickstart, then explain features or architecture. You can see that pattern clearly in the READMEs for Streamlit, Chroma, and Docling. citeturn319240search1turn319240search4turn907798search1

If this README grows too large, GitHub recommends using the repository wiki for longer-form documentation, and GitHub also truncates README content beyond 500 KiB on the repository page. citeturn907798search7turn907798search0

---

## Contributing

Contributions are welcome, but clarity matters more than activity.

If you want to contribute, do one of these:

- improve parsing fidelity
- improve chunk-quality evaluation
- reduce token and latency cost
- harden metadata traceability
- build the retrieval and UI layers without breaking the ingestion logic

Do not submit cosmetic complexity disguised as progress.

---

## Final note

This repository is based on a simple belief:

> If you want better answers, start by building better chunks.

Everything else comes later.

