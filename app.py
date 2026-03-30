import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field


# ==========================================================
# 1. Secret only from .env
# ==========================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing in .env")


# ==========================================================
# 2. Plain Python config
# ==========================================================
EMBED_MODEL = "text-embedding-3-large"
CONTROL_MODEL = "gpt-5.4-mini"
SUBANSWER_MODEL = "gpt-5.4"
FINAL_MODEL = "gpt-5.4"

CHROMA_PATH = "vector_db/chroma_db"
CHROMA_COLLECTION_NAME = "embeddings_db"
PAGE_IMAGES_DIR = Path("sample_doc_assets/page_images")

TOP_K = 4
SUMMARY_BATCH_SIZE = 3
RAW_MEMORY_WINDOW = 2


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
def get_control_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=CONTROL_MODEL,
        temperature=0,
    )


@st.cache_resource
def get_subanswer_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=SUBANSWER_MODEL,
        temperature=0,
    )


@st.cache_resource
def get_final_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=FINAL_MODEL,
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
control_llm = get_control_llm()
subanswer_llm = get_subanswer_llm()
final_llm = get_final_llm()
collection = get_collection()


# ==========================================================
# 4. Utility helpers
# ==========================================================
def safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def model_dump_compat(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


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


def format_pages_for_sources(pages: List[int]) -> str:
    unique_pages = sorted(set(int(page) for page in pages if isinstance(page, int)))
    if not unique_pages:
        return "None"
    return ", ".join(str(page) for page in unique_pages)


def strip_sources_line(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    lines = cleaned.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()

    if lines and lines[-1].strip().lower().startswith("sources used:"):
        lines.pop()

    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines).strip()


def format_answer_with_sources(answer_markdown: str, source_pages: List[int]) -> str:
    body = strip_sources_line(answer_markdown)
    if not body:
        body = "I do not know based on the retrieved context."
    return f"{body}\n\nSources used: {format_pages_for_sources(source_pages)}"


def collect_result_pages(results: List[Dict[str, Any]]) -> List[int]:
    pages: List[int] = []
    for item in results:
        for chunk in item.get("chunks", []):
            pages.extend(chunk.get("pages", []))
    return sorted(set(pages))


def collect_supported_pages(results: List[Dict[str, Any]]) -> List[int]:
    pages: List[int] = []
    for item in results:
        if item.get("supported"):
            pages.extend(item.get("source_pages", []))
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
# 5. Chat state + rolling memory
# ==========================================================
def initialize_chat_state() -> None:
    defaults = {
        "messages": [],
        "completed_turns": [],
        "conversation_summary": "",
        "last_summarized_pair_index": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            if isinstance(value, list):
                st.session_state[key] = []
            else:
                st.session_state[key] = value


def clear_chat_state() -> None:
    st.session_state.messages = []
    st.session_state.completed_turns = []
    st.session_state.conversation_summary = ""
    st.session_state.last_summarized_pair_index = 0


def format_turns(turns: List[Dict[str, str]]) -> str:
    formatted: List[str] = []

    for idx, turn in enumerate(turns, start=1):
        formatted.append(
            f"Exchange {idx}\n"
            f"User: {turn.get('user', '').strip()}\n"
            f"Assistant: {turn.get('assistant', '').strip()}"
        )

    return "\n\n".join(formatted)


def build_memory_context(
    conversation_summary: str,
    completed_turns: List[Dict[str, str]],
    current_user_query: str,
) -> str:
    sections: List[str] = []
    recent_turns = completed_turns[-RAW_MEMORY_WINDOW:]

    if conversation_summary.strip():
        sections.append(
            "Rolling summary of earlier conversation:\n"
            f"{conversation_summary.strip()}"
        )

    if recent_turns:
        sections.append(
            f"Latest {len(recent_turns)} raw exchange(s):\n"
            f"{format_turns(recent_turns)}"
        )

    sections.append(f"Current user message:\n{current_user_query.strip()}")
    return "\n\n".join(sections)


def summarize_memory_batch(
    existing_summary: str,
    batch_turns: List[Dict[str, str]],
) -> Optional[str]:
    if not batch_turns:
        return existing_summary.strip()

    payload = {
        "existing_summary": existing_summary.strip(),
        "new_exchanges": batch_turns,
    }

    try:
        response = control_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Update a compact rolling text memory for a PDF chat assistant.\n"
                        "Keep only details useful for future follow-up questions.\n"
                        "Capture discussed topics, facts already stated, unresolved follow-ups, and explicit uncertainty.\n"
                        "Use only the provided text conversation content.\n"
                        "Do not add new facts, guesses, or citations.\n"
                        "Return plain text only."
                    )
                ),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
            ]
        )
        text = str(response.content).strip()
        return text or existing_summary.strip()
    except Exception:
        fallback_parts: List[str] = []

        if existing_summary.strip():
            fallback_parts.append(existing_summary.strip())

        for turn in batch_turns:
            fallback_parts.append(f"User asked: {turn.get('user', '').strip()}")
            fallback_parts.append(
                f"Assistant answered: {turn.get('assistant', '').strip()}"
            )

        fallback = "\n".join(fallback_parts).strip()
        return fallback[-3000:] if fallback else existing_summary.strip()


def update_conversation_summary_if_needed() -> None:
    start_idx = st.session_state.last_summarized_pair_index
    completed_turns = st.session_state.completed_turns

    while len(completed_turns) - start_idx >= SUMMARY_BATCH_SIZE:
        end_idx = start_idx + SUMMARY_BATCH_SIZE
        batch = completed_turns[start_idx:end_idx]
        updated_summary = summarize_memory_batch(
            existing_summary=st.session_state.conversation_summary,
            batch_turns=batch,
        )

        if updated_summary is None:
            break

        st.session_state.conversation_summary = updated_summary
        start_idx = end_idx

    st.session_state.last_summarized_pair_index = start_idx


# ==========================================================
# 6. Query analysis
# ==========================================================
class RetrievalTask(BaseModel):
    sub_question: str = Field(
        description="Standalone content question to answer from the PDF."
    )
    retrieval_query: str = Field(
        description="Retrieval-only search query focused on topic terms, without audience, tone, or formatting instructions."
    )


class QueryPlan(BaseModel):
    standalone_query: str = Field(
        description="Latest user message rewritten as a standalone question using text memory only."
    )
    answer_instructions: str = Field(
        description="How the answer should be phrased for the user, including audience, tone, detail level, brevity, structure, and teaching style."
    )
    sub_queries: List[str] = Field(
        description="Explicit list of content sub-queries derived from the user request. Use one item unless the user clearly asked multiple independent questions."
    )
    retrieval_tasks: List[RetrievalTask] = Field(
        description="Retrieval tasks aligned with sub_queries and ordered the same way."
    )


class SplitTaskPlan(BaseModel):
    retrieval_tasks: List[RetrievalTask] = Field(
        description="Independent retrieval tasks extracted from one composite user query."
    )


def looks_multi_part_query(text: str) -> bool:
    lowered = text.lower().strip()
    if not lowered:
        return False

    if lowered.count(" and ") >= 2:
        return True

    if (" and " in lowered or ";" in lowered or "," in lowered) and len(lowered) > 90:
        return True

    question_markers = [
        "what is",
        "what are",
        "how does",
        "how do",
        "why does",
        "why do",
        "give an example",
        "example",
        "math behind",
        "formula",
        "equation",
        "compare",
        "difference between",
    ]
    marker_hits = sum(1 for marker in question_markers if marker in lowered)
    return marker_hits >= 2


def maybe_expand_multi_part_tasks(
    standalone_query: str,
    answer_instructions: str,
    tasks: List[RetrievalTask],
) -> List[RetrievalTask]:
    if len(tasks) != 1 or not looks_multi_part_query(standalone_query):
        return tasks

    splitter_llm = control_llm.with_structured_output(SplitTaskPlan)

    try:
        result = splitter_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Split one composite user query into independent retrieval tasks only when the query clearly asks about multiple distinct topics.\n"
                        "Return retrieval_tasks only.\n\n"
                        "Rules:\n"
                        "- Split when one sentence asks about multiple concepts, mechanisms, examples, formulas, or architectures.\n"
                        "- Keep tightly connected phrases together in the same task.\n"
                        "- Keep audience/style instructions out of retrieval_query.\n"
                        "- Prefer 2 to 4 tasks when splitting is warranted.\n"
                        "- If the query is actually one topic, return exactly one task.\n\n"
                        "Example:\n"
                        "Query: What are encoder-decoder models and the attention mechanism (with an example), and what is the basic math behind attention explained simply?\n"
                        "Tasks:\n"
                        "1. What are encoder-decoder models? | retrieval_query: encoder-decoder models architecture\n"
                        "2. What is the attention mechanism? Give an example. | retrieval_query: attention mechanism example\n"
                        "3. What is the basic math behind attention? | retrieval_query: attention math formula queries keys values softmax"
                    )
                ),
                HumanMessage(
                    content=(
                        f"Standalone query:\n{standalone_query}\n\n"
                        f"Answer instructions:\n{answer_instructions or 'None'}\n\n"
                        f"Current single task:\n{json.dumps([model_dump_compat(task) for task in tasks], ensure_ascii=False, indent=2)}"
                    )
                ),
            ]
        )

        expanded_tasks: List[RetrievalTask] = []
        for task in result.retrieval_tasks:
            sub_question = task.sub_question.strip()
            retrieval_query = task.retrieval_query.strip()
            if not sub_question:
                continue
            expanded_tasks.append(
                RetrievalTask(
                    sub_question=sub_question,
                    retrieval_query=retrieval_query or sub_question,
                )
            )

        return expanded_tasks if len(expanded_tasks) > 1 else tasks
    except Exception:
        return tasks


def analyze_user_query(
    raw_query: str,
    conversation_summary: str,
    completed_turns: List[Dict[str, str]],
) -> QueryPlan:
    cleaned_query = raw_query.strip()
    fallback_task = RetrievalTask(
        sub_question=cleaned_query,
        retrieval_query=cleaned_query,
    )
    fallback_plan = QueryPlan(
        standalone_query=cleaned_query,
        answer_instructions="",
        sub_queries=[cleaned_query] if cleaned_query else [],
        retrieval_tasks=[fallback_task] if cleaned_query else [],
    )

    if not cleaned_query:
        return fallback_plan

    memory_context = build_memory_context(
        conversation_summary=conversation_summary,
        completed_turns=completed_turns,
        current_user_query=cleaned_query,
    )
    planner_llm = control_llm.with_structured_output(QueryPlan)

    try:
        result = planner_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Plan a grounded PDF QA request using text memory only.\n"
                        "You must separate retrieval intent from answer style.\n\n"
                        "Return:\n"
                        "- standalone_query: resolve follow-up references using memory, but keep the meaning unchanged\n"
                        "- answer_instructions: how to explain the answer to the user\n"
                        "- sub_queries: explicit content questions only\n"
                        "- retrieval_tasks: retrieval-only search text for each sub-query\n\n"
                        "Rules:\n"
                        "- Split only when the user truly asks multiple independent questions.\n"
                        "- If the request is one topic plus style constraints, return exactly one sub-query.\n"
                        "- Keep style constraints such as audience, tone, detail level, bullets, or step-by-step in answer_instructions, not retrieval_query.\n"
                        "- Keep technical terms intact.\n"
                        "- sub_queries and retrieval_tasks must have matching order and count.\n"
                        "- Do not answer the question."
                    )
                ),
                HumanMessage(content=memory_context),
            ]
        )

        if not result.retrieval_tasks:
            return fallback_plan

        normalized_tasks: List[RetrievalTask] = []
        for task in result.retrieval_tasks:
            sub_question = task.sub_question.strip() or result.standalone_query.strip() or cleaned_query
            retrieval_query = task.retrieval_query.strip() or sub_question
            normalized_tasks.append(
                RetrievalTask(
                    sub_question=sub_question,
                    retrieval_query=retrieval_query,
                )
            )

        normalized_tasks = maybe_expand_multi_part_tasks(
            standalone_query=result.standalone_query.strip() or cleaned_query,
            answer_instructions=result.answer_instructions.strip(),
            tasks=normalized_tasks,
        )
        normalized_sub_queries = [task.sub_question for task in normalized_tasks]
        return QueryPlan(
            standalone_query=result.standalone_query.strip() or cleaned_query,
            answer_instructions=result.answer_instructions.strip(),
            sub_queries=normalized_sub_queries,
            retrieval_tasks=normalized_tasks,
        )
    except Exception:
        return fallback_plan


# ==========================================================
# 7. Retrieval
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
# 8. Answer generation
# ==========================================================
class GroundedSubAnswer(BaseModel):
    supported: bool = Field(
        description="Whether every claim in the answer is directly supported by the supplied evidence."
    )
    answer_markdown: str = Field(
        description="Grounded answer body only, without the final Sources used line."
    )
    source_pages: List[int] = Field(
        description="Subset of allowed pages actually used for the answer."
    )


class VerificationResult(BaseModel):
    had_unsupported_claims: bool = Field(
        description="True if the candidate answer contained unsupported claims or incorrect source attribution."
    )
    corrected_answer_markdown: str = Field(
        description="Corrected final answer with exactly one final Sources used line."
    )
    issues: List[str] = Field(
        description="Short notes describing unsupported claims or source issues found during verification."
    )


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


def build_results_evidence_context(results: List[Dict[str, Any]]) -> str:
    sections: List[str] = []

    for idx, item in enumerate(results, start=1):
        sections.append(
            f"SUB_QUERY_{idx}\n"
            f"question: {item.get('sub_question', '')}\n"
            f"retrieval_query: {item.get('retrieval_query', '')}\n"
            f"supported: {item.get('supported')}\n"
            f"source_pages: {format_pages_for_sources(item.get('source_pages', []))}\n"
            f"answer_markdown:\n{item.get('answer_markdown', '')}\n\n"
            f"retrieved_context:\n{build_chunk_context(item.get('chunks', []))}"
        )

    return "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(sections)


def answer_subquestion(
    sub_question: str,
    answer_instructions: str,
    chunks: List[Dict[str, Any]],
    page_images: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not chunks:
        answer_markdown = "I do not know based on the retrieved context."
        return {
            "supported": False,
            "answer_markdown": answer_markdown,
            "source_pages": [],
            "answer": format_answer_with_sources(answer_markdown, []),
        }

    context_text = build_chunk_context(chunks)
    pages_available = sorted(
        {
            page
            for chunk in chunks
            for page in chunk.get("pages", [])
        }
    )
    style_instructions = (
        answer_instructions.strip()
        or "No extra style instructions. Give a clear and complete answer."
    )

    prompt_text = (
        "You are producing one grounded sub-answer for a PDF QA system.\n\n"
        "You are given:\n"
        "1) one content question\n"
        "2) answer instructions from the user\n"
        "3) retrieved text chunks\n"
        "4) page screenshots from the same allowed pages\n\n"
        "Rules:\n"
        "- Every claim must be directly supported by the chunks or directly readable in the screenshots from the allowed pages.\n"
        "- Screenshots are supporting evidence only; do not infer extra facts beyond what is visible.\n"
        "- Follow the answer instructions closely.\n"
        "- Keep inline citations accurate using [p. X] or [pp. X, Y].\n"
        "- If formulas or equations matter, format them in readable Markdown with display math blocks using $$...$$ when helpful.\n"
        "- answer_markdown must NOT include the final Sources used line.\n"
        "- If the evidence is insufficient, set supported=false, answer_markdown='I do not know based on the retrieved context.', and source_pages=[].\n"
        "- source_pages must be a subset of the allowed pages.\n\n"
        f"Content question:\n{sub_question}\n\n"
        f"Answer instructions:\n{style_instructions}\n\n"
        f"Allowed pages:\n{pages_available if pages_available else 'None'}\n\n"
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
        structured_llm = subanswer_llm.with_structured_output(GroundedSubAnswer)
        result = structured_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Return a strictly grounded sub-answer. "
                        "Do not guess. "
                        "If support is partial or uncertain, be conservative."
                    )
                ),
                HumanMessage(content=human_content),
            ]
        )

        source_pages = [
            page for page in parse_pages(result.source_pages) if page in pages_available
        ]
        answer_markdown = strip_sources_line(result.answer_markdown)
        if not result.supported:
            answer_markdown = "I do not know based on the retrieved context."
            source_pages = []

        return {
            "supported": bool(result.supported),
            "answer_markdown": answer_markdown,
            "source_pages": source_pages,
            "answer": format_answer_with_sources(answer_markdown, source_pages),
        }
    except Exception:
        answer_markdown = "I do not know based on the retrieved context."
        return {
            "supported": False,
            "answer_markdown": answer_markdown,
            "source_pages": [],
            "answer": format_answer_with_sources(answer_markdown, []),
        }


def synthesize_final_answer_candidate(
    resolved_query: str,
    answer_instructions: str,
    results: List[Dict[str, Any]],
    page_images: List[Dict[str, Any]],
) -> str:
    supported_results = [item for item in results if item.get("supported")]
    if not supported_results:
        return "I do not know based on the retrieved context.\n\nSources used: None"

    supported_payload = []
    for item in supported_results:
        supported_payload.append(
            {
                "sub_question": item["sub_question"],
                "answer_markdown": item["answer_markdown"],
                "source_pages": item["source_pages"],
            }
        )

    allowed_pages = collect_result_pages(results)
    preferred_pages = collect_supported_pages(supported_results) or allowed_pages
    evidence_context = build_results_evidence_context(supported_results)
    style_instructions = (
        answer_instructions.strip()
        or "No extra style instructions. Give a clear and complete answer."
    )

    prompt_text = (
        "You are producing the final grounded answer for a PDF QA system.\n\n"
        "You are given:\n"
        "1) the resolved user query\n"
        "2) answer instructions\n"
        "3) grounded sub-answers\n"
        "4) compact retrieved evidence\n"
        "5) screenshots from the same allowed pages\n\n"
        "Rules:\n"
        "- Use only the grounded sub-answers and supporting evidence.\n"
        "- Do not add any new facts.\n"
        "- Respect the answer instructions.\n"
        "- Keep inline citations accurate.\n"
        "- If formulas or equations are relevant, preserve them in readable Markdown or $$...$$ blocks.\n"
        "- End with exactly one final line: Sources used: <page numbers only or None>.\n"
        "- The final Sources used line must use only these pages: "
        f"{format_pages_for_sources(preferred_pages)}.\n\n"
        f"Resolved user query:\n{resolved_query}\n\n"
        f"Answer instructions:\n{style_instructions}\n\n"
        f"Grounded sub-answers:\n{json.dumps(supported_payload, ensure_ascii=False, indent=2)}\n\n"
        f"Evidence context:\n{evidence_context}"
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
        response = final_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Produce one final grounded answer only from the supplied material. "
                        "Do not invent facts or citations."
                    )
                ),
                HumanMessage(content=human_content),
            ]
        )
        return str(response.content).strip()
    except Exception:
        fallback_body = "\n\n".join(item["answer_markdown"] for item in supported_results)
        return format_answer_with_sources(fallback_body, preferred_pages)


def verify_final_answer(
    candidate_answer: str,
    resolved_query: str,
    answer_instructions: str,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    allowed_pages = collect_result_pages(results)
    supported_results = [
        {
            "sub_question": item["sub_question"],
            "supported": item["supported"],
            "answer_markdown": item["answer_markdown"],
            "source_pages": item["source_pages"],
        }
        for item in results
    ]
    evidence_context = build_results_evidence_context(results)
    style_instructions = (
        answer_instructions.strip()
        or "No extra style instructions. Give a clear and complete answer."
    )

    try:
        verifier_llm = control_llm.with_structured_output(VerificationResult)
        result = verifier_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Verify whether a candidate final answer is fully grounded in the supplied evidence.\n"
                        "If there are unsupported claims, incorrect implications, or invalid source pages, rewrite the answer conservatively using only supported content.\n"
                        "The corrected answer must contain exactly one final line in the form 'Sources used: ...'.\n"
                        "Use only allowed source pages.\n"
                        "If nothing is sufficiently supported, return 'I do not know based on the retrieved context.' followed by 'Sources used: None'."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Resolved user query:\n{resolved_query}\n\n"
                        f"Answer instructions:\n{style_instructions}\n\n"
                        f"Allowed source pages:\n{format_pages_for_sources(allowed_pages)}\n\n"
                        f"Grounded sub-answers:\n{json.dumps(supported_results, ensure_ascii=False, indent=2)}\n\n"
                        f"Evidence context:\n{evidence_context}\n\n"
                        f"Candidate final answer:\n{candidate_answer}"
                    )
                ),
            ]
        )

        corrected = str(result.corrected_answer_markdown).strip()
        if not corrected:
            corrected = candidate_answer.strip()

        return {
            "final_answer": corrected,
            "had_unsupported_claims": bool(result.had_unsupported_claims),
            "verification_issues": result.issues or [],
        }
    except Exception:
        return {
            "final_answer": candidate_answer.strip(),
            "had_unsupported_claims": False,
            "verification_issues": [],
        }


# ==========================================================
# 9. Full pipeline
# ==========================================================
def run_pipeline(
    raw_query: str,
    conversation_summary: str,
    completed_turns: List[Dict[str, str]],
) -> Dict[str, Any]:
    memory_context = build_memory_context(
        conversation_summary=conversation_summary,
        completed_turns=completed_turns,
        current_user_query=raw_query,
    )
    query_plan = analyze_user_query(
        raw_query=raw_query,
        conversation_summary=conversation_summary,
        completed_turns=completed_turns,
    )

    results: List[Dict[str, Any]] = []
    all_images: List[Dict[str, Any]] = []

    for task in query_plan.retrieval_tasks:
        retrieval_query = task.retrieval_query.strip() or task.sub_question.strip()
        chunks = retrieve_chunks(retrieval_query)
        page_images = load_page_images_from_chunks(chunks)
        subanswer = answer_subquestion(
            sub_question=task.sub_question,
            answer_instructions=query_plan.answer_instructions,
            chunks=chunks,
            page_images=page_images,
        )

        results.append(
            {
                "sub_question": task.sub_question,
                "retrieval_query": retrieval_query,
                "chunks": chunks,
                "page_images": page_images,
                "supported": subanswer["supported"],
                "answer_markdown": subanswer["answer_markdown"],
                "source_pages": subanswer["source_pages"],
                "answer": subanswer["answer"],
            }
        )
        all_images.extend(page_images)

    final_images = dedupe_images_keep_order(all_images)
    candidate_final_answer = synthesize_final_answer_candidate(
        resolved_query=query_plan.standalone_query,
        answer_instructions=query_plan.answer_instructions,
        results=results,
        page_images=final_images,
    )
    verification = verify_final_answer(
        candidate_answer=candidate_final_answer,
        resolved_query=query_plan.standalone_query,
        answer_instructions=query_plan.answer_instructions,
        results=results,
    )

    return {
        "raw_query": raw_query,
        "resolved_query": query_plan.standalone_query,
        "answer_instructions": query_plan.answer_instructions,
        "sub_queries": query_plan.sub_queries,
        "retrieval_tasks": [model_dump_compat(task) for task in query_plan.retrieval_tasks],
        "memory_context": memory_context,
        "summary_snapshot": conversation_summary,
        "results": results,
        "candidate_final_answer": candidate_final_answer,
        "verification_issues": verification["verification_issues"],
        "had_unsupported_claims": verification["had_unsupported_claims"],
        "final_answer": verification["final_answer"],
        "all_images": final_images,
    }


# ==========================================================
# 10. Rendering helpers
# ==========================================================
def render_source_images(images: List[Dict[str, Any]]) -> None:
    if not images:
        st.caption("No source page images were found.")
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


def render_debug(bundle: Dict[str, Any], key_prefix: str) -> None:
    st.markdown("### Query resolution")
    st.write(f"**Raw query:** {bundle['raw_query']}")
    st.write(f"**Resolved query:** {bundle['resolved_query']}")
    st.write(
        f"**Answer instructions:** {bundle.get('answer_instructions') or 'None'}"
    )

    if bundle.get("summary_snapshot"):
        st.markdown("### Rolling summary used")
        st.write(bundle["summary_snapshot"])

    if bundle.get("memory_context"):
        st.markdown("### Memory context sent to the planner")
        st.text_area(
            label="Memory context",
            value=bundle["memory_context"],
            height=220,
            key=f"{key_prefix}_memory_context",
            disabled=True,
        )

    if bundle.get("sub_queries"):
        st.markdown("### Sub-queries")
        for idx, sub_query in enumerate(bundle["sub_queries"], start=1):
            st.write(f"{idx}. {sub_query}")

    if bundle.get("retrieval_tasks"):
        st.markdown("### Retrieval plan")
        for idx, task in enumerate(bundle["retrieval_tasks"], start=1):
            st.write(f"{idx}. **Question:** {task['sub_question']}")
            st.write(f"   **Retrieval query:** {task['retrieval_query']}")

    st.markdown("### Retrieved Sources By Sub-query")
    for idx, item in enumerate(bundle["results"], start=1):
        st.markdown(f"#### Sub-question {idx}")
        st.write(f"**Question:** {item['sub_question']}")
        st.write(f"**Retrieval query:** {item['retrieval_query']}")
        retrieved_pages = sorted(
            {
                page
                for chunk in item.get("chunks", [])
                for page in chunk.get("pages", [])
            }
        )
        st.write(
            f"**Retrieved pages:** {format_pages_for_sources(retrieved_pages)}"
        )

        if not item["chunks"]:
            st.warning("No chunks retrieved.")
        else:
            for chunk in item["chunks"]:
                pages_text = ", ".join(str(p) for p in chunk.get("pages", [])) or "unknown"
                sim = chunk.get("similarity")
                sim_text = f"{sim:.4f}" if isinstance(sim, float) else "N/A"

                st.write(
                    f"- ID: `{chunk['id']}` | pages: {pages_text} | similarity: {sim_text}"
                )


def render_assistant_bundle(bundle: Dict[str, Any], key_prefix: str) -> None:
    with st.expander("Sources and debug", expanded=False):
        render_source_images(bundle.get("all_images", []))
        render_debug(bundle, key_prefix=key_prefix)


def render_chat_history() -> None:
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("bundle"):
                render_assistant_bundle(
                    bundle=message["bundle"],
                    key_prefix=f"history_{idx}",
                )


# ==========================================================
# 11. Streamlit app
# ==========================================================
def main() -> None:
    st.set_page_config(page_title="PDF Retrieval QA Chat", page_icon=":books:", layout="wide")
    initialize_chat_state()

    header_col, action_col = st.columns([6, 1])
    with header_col:
        st.title("PDF Retrieval QA Chat")
        st.write(
            "Flow: text memory -> query planning -> grounded sub-answers -> multimodal final answer -> verification"
        )
    with action_col:
        st.write("")
        st.write("")
        if st.button("Clear chat", use_container_width=True):
            clear_chat_state()
            st.rerun()

    render_chat_history()

    user_query = st.chat_input(
        "Ask about the document. Style requests stay in the answer, not retrieval."
    )
    if not user_query:
        return

    cleaned_query = user_query.strip()
    if not cleaned_query:
        st.warning("Please enter a question.")
        return

    if collection.count() == 0:
        st.error(
            f"Collection '{CHROMA_COLLECTION_NAME}' at '{CHROMA_PATH}' is empty. "
            "Fix that first."
        )
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": cleaned_query,
        }
    )

    with st.chat_message("user"):
        st.markdown(cleaned_query)

    with st.chat_message("assistant"):
        with st.spinner("Running retrieval pipeline..."):
            bundle = run_pipeline(
                raw_query=cleaned_query,
                conversation_summary=st.session_state.conversation_summary,
                completed_turns=st.session_state.completed_turns,
            )

        assistant_reply = bundle["final_answer"]
        st.markdown(assistant_reply)
        render_assistant_bundle(
            bundle=bundle,
            key_prefix=f"live_{len(st.session_state.messages)}",
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_reply,
            "bundle": bundle,
        }
    )
    st.session_state.completed_turns.append(
        {
            "user": cleaned_query,
            "assistant": assistant_reply,
        }
    )
    update_conversation_summary_if_needed()


if __name__ == "__main__":
    main()
