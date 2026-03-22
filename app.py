import os
import json
import re
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional
from typing import List
from pydantic import BaseModel, Field


import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ==========================================================
# 1. Secret only from .env
# ==========================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing in .env")

# ==========================================================
# 2. Plain Python config
# Change these if your local folders / collection names differ
# ==========================================================
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-5"

CHROMA_PATH = "vector_db/chroma_db"
CHROMA_COLLECTION_NAME = "embeddings_db"   # change to "embedding" if that is your real non-empty collection
PAGE_IMAGES_DIR = Path("sample_doc_assets/page_images")

TOP_K = 4


# ==========================================================
# 3. Cached resources
# ==========================================================
@st.cache_resource
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model=EMBED_MODEL,
    )


@st.cache_resource
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=CHAT_MODEL,
        temperature=0.1,
    )


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    except Exception as e:
        available = []
        try:
            available = [
                c.name if hasattr(c, "name") else str(c)
                for c in client.list_collections()
            ]
        except Exception:
            pass
        raise RuntimeError(
            f"Could not open collection '{CHROMA_COLLECTION_NAME}' from '{CHROMA_PATH}'. "
            f"Available collections: {available}"
        ) from e

    return collection


embeddings = get_embeddings()
llm = get_llm()
collection = get_collection()


# ==========================================================
# 4. Utility helpers
# ==========================================================
def safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def parse_pages(value: Any) -> List[int]:
    if value is None:
        return []

    if isinstance(value, int):
        return [value]

    if isinstance(value, float):
        return [int(value)]

    if isinstance(value, list):
        pages = []
        for item in value:
            try:
                pages.append(int(item))
            except Exception:
                pass
        return sorted(set(pages))

    text = str(value).strip()
    if not text:
        return []

    parsed = safe_json_loads(text)
    if parsed is not None and parsed != text:
        return parse_pages(parsed)

    pages: List[int] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if part.isdigit():
            pages.append(int(part))

    return sorted(set(pages))


def encode_image(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def find_page_image(page_no: int) -> Optional[Path]:
    direct = PAGE_IMAGES_DIR / f"page-{page_no}.png"
    if direct.exists():
        return direct

    if PAGE_IMAGES_DIR.exists():
        matches = sorted(PAGE_IMAGES_DIR.rglob(f"page-{page_no}.png"))
        if matches:
            return matches[0]

    parent = PAGE_IMAGES_DIR.parent
    if parent.exists():
        matches = sorted(parent.rglob(f"page-{page_no}.png"))
        if matches:
            return matches[0]

    return None


def dedupe_images_keep_order(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_pages = set()
    out = []
    for image in images:
        page = image.get("page")
        if page in seen_pages:
            continue
        seen_pages.add(page)
        out.append(image)
    return out



# ==========================================================
# 5. Query decomposition + rephrasing
# ==========================================================

class SplitQuestions(BaseModel):
    sub_questions: List[str] = Field(
        description="Atomic sub-questions extracted from the user query"
    )


def split_user_query(user_query: str) -> List[str]:
    splitter_llm = llm.with_structured_output(SplitQuestions)

    try:
        result = splitter_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Split the user's input into atomic sub-questions only when it clearly contains multiple asks.\n"
                        "If it is really one question, return exactly 1.\n"
                        "Never return more than 5 sub-questions.\n\n"
                        "Rules:\n"
                        "- Preserve the original meaning.\n"
                        "- Do not invent new questions.\n"
                        "- Do not over-split a single concept into many parts.\n"
                        "- If two phrases belong to the same concept, keep them together.\n"
                        "- Keep each sub-question short and standalone.\n"
                        "- question_count must exactly match the number of returned sub_questions.\n"
                        "- Prefer fewer questions when uncertain."
                    )
                ),
                HumanMessage(content=user_query),
            ]
        )

        items = [q.strip() for q in result.sub_questions if q.strip()]

        # # hard safety cap
        # if not items:
        #     return [user_query]

        # items = items[:5]

        # # if model says 1, trust that and keep only first
        # if getattr(result, "question_count", None) == 1:
        #     return [items[0]]

        # # keep count aligned
        # expected = max(1, min(5, int(getattr(result, "question_count", len(items)))))
        # items = items[:expected]

        return items if items else [user_query]

    except Exception:
        return [user_query]


def rephrase_query(sub_question: str) -> str:
    prompt = [
        SystemMessage(
            content=(
                "Rewrite the user's question into one short retrieval-friendly search query. "
                "Keep the meaning the same. Keep technical terms. Return only the rewritten query."
            )
        ),
        HumanMessage(content=sub_question),
    ]

    response = llm.invoke(prompt)
    text = str(response.content).strip()
    return text or sub_question



# ==========================================================
# 6. Retrieval
# ==========================================================
def retrieve_chunks(search_query: str) -> List[Dict[str, Any]]:
    query_vector = embeddings.embed_query(search_query)

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    ids = result.get("ids", [[]])[0]

    chunks: List[Dict[str, Any]] = []
    for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        metadata = metadata or {}
        pages = parse_pages(metadata.get("pages", metadata.get("source")))

        similarity = None
        if distance is not None:
            similarity = 1 - float(distance)

        chunks.append(
            {
                "id": str(chunk_id),
                "text": document or "",
                "metadata": metadata,
                "pages": pages,
                "similarity": similarity,
            }
        )

    return chunks


def load_page_images_from_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pages: List[int] = []
    for chunk in chunks:
        pages.extend(chunk.get("pages", []))

    images: List[Dict[str, Any]] = []
    for page_no in sorted(set(pages)):
        image_path = find_page_image(page_no)
        if not image_path:
            continue

        image_b64 = encode_image(image_path)
        if not image_b64:
            continue

        images.append(
            {
                "page": page_no,
                "path": str(image_path),
                "b64": image_b64,
            }
        )

    return images


# ==========================================================
# 7. Answer generation
# ==========================================================
def build_chunk_context(chunks: List[Dict[str, Any]]) -> str:
    parts: List[str] = []

    for idx, chunk in enumerate(chunks, start=1):
        pages = chunk.get("pages", [])
        pages_text = ", ".join(str(p) for p in pages) if pages else "unknown"
        sim = chunk.get("similarity")
        sim_text = f"{sim:.4f}" if isinstance(sim, float) else "N/A"

        parts.append(
            f"CHUNK_{idx}\n"
            f"id: {chunk.get('id')}\n"
            f"pages: {pages_text}\n"
            f"similarity: {sim_text}\n"
            f"text:\n{chunk.get('text', '')}"
        )

    return "\n\n" + ("\n\n" + "-" * 80 + "\n\n").join(parts)


def answer_subquestion(
    sub_question: str,
    chunks: List[Dict[str, Any]],
    page_images: List[Dict[str, Any]],
) -> str:
    if not chunks:
        return "I do not know based on the retrieved context.\n\nSources used: None"

    context_text = build_chunk_context(chunks)
    pages_available = sorted(
        {
            page
            for chunk in chunks
            for page in chunk.get("pages", [])
        }
    )

    prompt_text = (
        "You are a grounded multimodal RAG assistant for one PDF.\n\n"
        "You are given:\n"
        "1) one user sub-question\n"
        "2) retrieved text chunks from ChromaDB\n"
        "3) page screenshots from the same source pages\n\n"
        "Rules:\n"
        "- Answer ONLY from the supplied evidence.\n"
        "- If the answer is not supported, say you do not know based on the retrieved context.\n"
        "- Be clear and direct.\n"
        "- Keep the response short and compact while preserving all key technical information.\n"
        "- Do not omit important details, but avoid repetition and unnecessary explanation.\n"
        "- Prefer dense, information-rich sentences over long explanations.\n"
        "- Cite supporting pages inline like [p. 4] or [pp. 4, 5].\n"
        "- End with exactly one line: Sources used: <page numbers only or None>\n"
        "- Do not invent page numbers.\n\n"
        f"Sub-question:\n{sub_question}\n\n"
        f"Pages available from chunks/images:\n{pages_available if pages_available else 'None'}\n\n"
        f"Retrieved context:\n{context_text}"
    )

    human_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    for image in page_images:
        human_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image['b64']}"},
            }
        )

    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Answer only from supplied evidence. "
                        "Do not hallucinate. Keep page citations correct."
                    )
                ),
                HumanMessage(content=human_content),
            ]
        )
        return str(response.content).strip()
    except Exception:
        return "I do not know based on the retrieved context.\n\nSources used: None"


def synthesize_final_answer(original_query: str, results: List[Dict[str, Any]]) -> str:
    if not results:
        return "I do not know based on the retrieved context."

    if len(results) == 1:
        return results[0]["answer"]

    payload = []
    for i, item in enumerate(results, start=1):
        payload.append(
            {
                "sub_question": item["sub_question"],
                "answer": item["answer"],
            }
        )

    try:
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are given answers to multiple sub-questions from the same PDF.\n"
                        "Merge them into one final clean response.\n"
                        "Rules:\n"
                        "- Preserve factual content and page citations.\n"
                        "- Do not add any new facts.\n"
                        "- Remove repetition.\n"
                        "- If the parts are connected, make the flow natural.\n"
                        "- Keep the final response readable."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Original user query:\n{original_query}\n\n"
                        f"Sub-answer bundle:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
                    )
                ),
            ]
        )
        return str(response.content).strip()
    except Exception:
        merged = []
        for item in results:
            merged.append(f"### {item['sub_question']}\n{item['answer']}")
        return "\n\n".join(merged)


# ==========================================================
# 8. Full pipeline
# ==========================================================
def run_pipeline(user_query: str) -> Dict[str, Any]:
    sub_questions = split_user_query(user_query)

    results: List[Dict[str, Any]] = []
    all_images: List[Dict[str, Any]] = []

    for sub_question in sub_questions:
        chunks = retrieve_chunks(sub_question)
        page_images = load_page_images_from_chunks(chunks)
        answer = answer_subquestion(
            sub_question=sub_question,
            chunks=chunks,
            page_images=page_images,
        )

        results.append(
            {
                "sub_question": sub_question,
                "chunks": chunks,
                "page_images": page_images,
                "answer": answer,
            }
        )

        all_images.extend(page_images)

    final_answer = synthesize_final_answer(user_query, results)

    return {
        "original_query": user_query,
        "sub_questions": sub_questions,
        "results": results,
        "final_answer": final_answer,
        "all_images": dedupe_images_keep_order(all_images),
    }


# ==========================================================
# 9. Rendering helpers
# ==========================================================
def render_source_images(images: List[Dict[str, Any]]) -> None:
    if not images:
        st.warning("No source page images were found.")
        return

    st.markdown("### Source page images")
    cols = st.columns(min(3, len(images)))

    for i, image in enumerate(images):
        with cols[i % len(cols)]:
            st.image(
                base64.b64decode(image["b64"]),
                caption=f"Page {image['page']}",
                use_container_width=True,
            )
            st.caption(image["path"])


def render_debug(bundle: Dict[str, Any]) -> None:
    st.markdown("### Query decomposition")
    for i, subq in enumerate(bundle["sub_questions"], start=1):
        st.write(f"{i}. {subq}")

    st.markdown("### Per-question retrieval")
    for idx, item in enumerate(bundle["results"], start=1):
        st.markdown(f"#### Sub-question {idx}")
        st.write(f"**Question:** {item['sub_question']}")

        if not item["chunks"]:
            st.warning("No chunks retrieved.")
            continue

        for chunk in item["chunks"]:
            pages_text = ", ".join(str(p) for p in chunk.get("pages", [])) or "unknown"
            sim = chunk.get("similarity")
            sim_text = f"{sim:.4f}" if isinstance(sim, float) else "N/A"

            st.write(
                f"- ID: `{chunk['id']}` | pages: {pages_text} | similarity: {sim_text}"
            )
            st.text_area(
                label=f"Chunk text - {chunk['id']}",
                value=chunk.get("text", ""),
                height=180,
                key=f"{idx}_{chunk['id']}",
            )

        st.markdown("**Sub-answer**")
        st.write(item["answer"])


# ==========================================================
# 10. Streamlit app
# ==========================================================
def main() -> None:
    st.set_page_config(page_title="PDF Retrieval QA", page_icon="📘", layout="wide")

    st.title("📘 PDF Retrieval QA")
    st.write(
        "Flow: split query → rephrase each part → retrieve per part → answer per part → merge final answer"
    )

    user_query = st.text_area(
        "Ask your question",
        height=140,
        placeholder="Example: explain attention mechanism and encoder-decoder and what is temperature",
    )

    if st.button("Ask"):
        if not user_query.strip():
            st.warning("Please enter a question.")
            return

        if collection.count() == 0:
            st.error(
                f"Collection '{CHROMA_COLLECTION_NAME}' at '{CHROMA_PATH}' is empty. "
                "Fix that first."
            )
            return

        with st.spinner("Running retrieval pipeline..."):
            bundle = run_pipeline(user_query.strip())

        st.markdown("## Final answer")
        st.write(bundle["final_answer"])

        render_source_images(bundle["all_images"])

        with st.expander("Debug", expanded=False):
            render_debug(bundle)


if __name__ == "__main__":
    main()