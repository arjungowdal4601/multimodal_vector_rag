import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import chromadb  # modern Chroma client


# ---------------------------------------------------------
# 1. Load environment variables
# ---------------------------------------------------------
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

# ---------------------------------------------------------
# 2. Initialize OpenAI Embeddings
# ---------------------------------------------------------
embeddings = OpenAIEmbeddings(
    api_key=openai_key,
    model=EMBEDDING_MODEL,
)


def ingest_chunks_to_chroma(
    pdf_path: str,
) -> int:
    """
    Load all page-wise chunk JSONs for a PDF, embed with OpenAI,
    and upsert into a shared Chroma collection.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    chroma_path : str
        Folder path where shared Chroma DB will store its persistent data.
    collection_name : str
        Name of the Chroma collection.

    Returns
    -------
    int
        Number of chunks successfully inserted.
    """

    # ---------------------------------------------------------
    # 3. Derive doc assets + page-wise JSON folder
    # ---------------------------------------------------------
    pdf_path = Path(pdf_path)
    asset_root = Path(f"{pdf_path.stem}_doc_assets")
    page_json_dir = asset_root / "page_wise_json"
    chroma_path="vector_db/chroma_db"
    collection_name="embeddings_db"

    # ---------------------------------------------------------
    # 4. Initialize shared ChromaDB
    # ---------------------------------------------------------
    chroma_client = chromadb.PersistentClient(path=chroma_path)

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # ---------------------------------------------------------
    # 5. Prepare documents, metadatas, and ids
    # ---------------------------------------------------------
    documents = []
    metadatas = []
    ids = []

    page_json_files = sorted(page_json_dir.glob("page_*.json"))

    for page_json_file in page_json_files:
        with open(page_json_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        for chunk_id, data in chunks.items():
            # Prefer refreshed_content, fallback to content
            text = (data.get("refreshed_content") or data.get("content") or "").strip()
            source_pages = data.get("source", [])
            metadata_addition = data.get("metadata_addition", {})

            if not text:
                continue

            # Shared DB needs unique ids across PDFs
            unique_id = f"{pdf_path.stem}_{page_json_file.stem}_{chunk_id}"

            # Chroma metadata values must be scalar types
            if isinstance(source_pages, list):
                pages_str = ",".join(str(p) for p in source_pages)
            else:
                pages_str = str(source_pages)

            documents.append(text)
            ids.append(unique_id)

            metadatas.append(
                {
                    "pages": pages_str,
                    "pdf_name": str(metadata_addition.get("pdf_name", pdf_path.name)),
                    "doc_assets_path": str(
                        metadata_addition.get("doc_assets_path", str(asset_root))
                    ),
                    "page_json_file": page_json_file.name,
                    "chunk_id": str(chunk_id),
                }
            )

    if not documents:
        print("⚠️ No non-empty chunks found to embed. Nothing was inserted.")
        return 0

    # ---------------------------------------------------------
    # 6. Compute embeddings with OpenAI
    # ---------------------------------------------------------
    print("🔄 Generating embeddings from OpenAI...")
    vectors = embeddings.embed_documents(documents)
    print(f"✅ Generated {len(vectors)} embeddings for {len(documents)} documents.")

    # ---------------------------------------------------------
    # 7. Insert into ChromaDB
    # ---------------------------------------------------------
    collection.upsert(
        documents=documents,
        embeddings=vectors,
        metadatas=metadatas,
        ids=ids,
    )

    print("\n🎉 DONE — Chunks stored in ChromaDB successfully!")
    print(f"📦 Location on disk: {chroma_path}")
    print(f"📚 Collection name : {collection_name}")
    print(f"📄 PDF name        : {pdf_path.name}")
    print(f"🔢 Total chunks    : {len(documents)}")

    return len(documents)