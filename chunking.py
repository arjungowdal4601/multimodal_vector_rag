import os
import base64
import json
from pathlib import Path
from typing import Dict, List,Optional, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


VISION_MODEL_NAME = os.getenv("OPENAI_VISION_MODEL", "gpt-5.2")

vision_llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=VISION_MODEL_NAME,
)

# ==========================================================
# 5. LLM-based semantic chunking of markdown (page-aware)
# ==========================================================


def _get_page_image_b64(
    page_no: int,
    page_images_dir: Path,
) -> Optional[str]:
    """
    Return base64-encoded PNG for the given page, or None if not found.
    """
    img_path = page_images_dir / f"page-{page_no}.png"
    if not img_path.exists():
        return None
    with img_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _chunk_single_page_with_llm(
    target_page: Dict[str, Any],
    next_page: Optional[Dict[str, Any]],
    page_images_dir: Path,
) -> Dict[str, Any]:
    """
    Use OpenAI LLM (vision_llm) to chunk ONE target page.

    - target_page: {"page_number": int, "text": str}
    - next_page  : same dict or None (only for context, NOT chunked)

    Returns a dict like:
    {
      "chunk_1": {"content": "...", "refreshed_content": "...", "source": [1]},
      "chunk_2": {...},
      ...
    }
    """

    target_num: int = target_page["page_number"]
    target_text: str = target_page["text"]

    if next_page is not None:
        next_num: int = next_page["page_number"]
        next_text: str = next_page["text"]
    else:
        next_num = None
        next_text = ""

    # Encode page screenshots (if they exist)
    target_img_b64 = _get_page_image_b64(target_num, page_images_dir)
    next_img_b64 = (
        _get_page_image_b64(next_num, page_images_dir)
        if next_num is not None
        else None
    )

    # Big instruction text – tells the model EXACTLY how to chunk + JSON format
    instruction_text = f"""
You are an expert at splitting technical PDFs into semantically meaningful chunks
for a Retrieval-Augmented Generation system.

You are given:

1. The MARKDOWN content of a TARGET page from a PDF.
2. Optionally, the MARKDOWN content of the NEXT page (ONLY for context).
3. Screenshots of these pages (if available).

Your job is to chunk ONLY the TARGET page, but use the NEXT page and images
to understand if any text or idea clearly flows across the page boundary.

Chunking rules (VERY IMPORTANT):

- Think like a human reader. Group together text that belongs together:
  * headings with their immediate paragraphs
  * bullet lists with their label/intro
  * figures/tables with their captions and descriptive bullets
- Prefer coherent, medium-sized chunks that are individually understandable
  , but semantic coherence is more important than size.
- The NEXT page is ONLY for understanding cross-page flow. DO NOT create chunks
  that are purely from the NEXT page.
- If a chunk on the TARGET page clearly continues onto the NEXT page
  (for example, a sentence or list broken at the page break), you may bring
  the minimal necessary continuation text from the NEXT page into that chunk
  and then set its "source" metadata to [TARGET_PAGE, NEXT_PAGE].
- Otherwise, use "source": [TARGET_PAGE] for that chunk.
- The "source" field MUST ALWAYS be a LIST of integers (page numbers),
  even if it is a single page.

Output format (STRICT):

Return ONLY a single JSON object. No markdown formatting, no comments, no prose.

The JSON must look like this:

{{
  "chunk_1": {{
    "content": "original-ish text for this chunk from the target page (plus any minimal continuation if needed).",
    "refreshed_content": "the same information, but rephrased meaning fully it should give meaning as a single entity for better embeddings; do NOT drop technical details.",
    "source": [{target_num}]  OR  [{target_num}, {next_num if next_num is not None else "SECOND_PAGE_NUMBER"}]
  }},
  "chunk_2": {{
    "content": "...",
    "refreshed_content": "...",
    "source": [ ... ]
  }}
  // add as many chunks as needed
}}

TARGET_PAGE_NUMBER = {target_num}

TARGET_PAGE_MARKDOWN:
{target_text}

NEXT_PAGE_NUMBER = {next_num if next_num is not None else "null"}

NEXT_PAGE_MARKDOWN:
{next_text if next_text else "NONE"}
""".strip()

    # Build multimodal content for HumanMessage
    human_content: List[Dict[str, Any]] = [
        {"type": "text", "text": instruction_text}
    ]

    if target_img_b64 is not None:
        human_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{target_img_b64}"
                },
            }
        )

    if next_img_b64 is not None:
        human_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{next_img_b64}"
                },
            }
        )

    messages = [
        SystemMessage(
            content=(
                "You are a precise document chunking engine. "
                "You MUST respond with a single valid JSON object only."
            )
        ),
        HumanMessage(content=human_content),
    ]

    resp = vision_llm.invoke(messages)
    raw = resp.content.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        # If model ever slips, it's useful to see what it produced
        raise ValueError(
            f"LLM did not return valid JSON for page {target_num}.\nRaw output:\n{raw}"
        ) from e

    return data


def _get_page_md_map(page_md_dir: str | Path) -> Dict[int, Path]:
    page_md_dir = Path(page_md_dir)

    page_map = {
        int(path.stem.split("_")[-1]): path
        for path in sorted(page_md_dir.glob("page_*.md"))
    }

    if not page_map:
        raise ValueError(f"No page markdown files found in: {page_md_dir}")

    return page_map


def _read_page(page_map: Dict[int, Path], page_no: int) -> Dict[str, Any] | None:
    path = page_map.get(page_no)
    if path is None:
        return None

    text = path.read_text(encoding="utf-8").strip()
    lines = text.splitlines()

    if lines and lines[0].strip().startswith("--- PAGE"):
        text = "\n".join(lines[1:]).strip()

    return {
        "page_number": page_no,
        "text": text,
    }


def _save_page_json(page_json_path: Path, per_page_chunks: Dict[str, Any], fallback_page_no: int) -> None:
    cleaned_chunks = {}

    for idx, chunk_data in enumerate(per_page_chunks.values(), start=1):
        src = chunk_data.get("source")

        if not src:
            src = [fallback_page_no]
        elif isinstance(src, int):
            src = [src]
        elif isinstance(src, str):
            src = json.loads(src)

        cleaned_chunks[f"chunk_{idx}"] = {
            "content": chunk_data.get("content", ""),
            "refreshed_content": chunk_data.get("refreshed_content", ""),
            "source": src,
        }

    page_json_path.write_text(
        json.dumps(cleaned_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def chunk_markdown_with_llm(
    pdf_path: str | Path,
) -> Path:
    
    asset_root = Path(f"{Path(pdf_path).stem}_doc_assets")
    page_images_dir = asset_root / "page_images"
    page_md_dir = asset_root / "pages_md"
    page_json_dir = asset_root / "page_wise_json"

    page_json_dir.mkdir(parents=True, exist_ok=True)

    page_map = _get_page_md_map(page_md_dir)

    for page_no in sorted(page_map):
        target_page = _read_page(page_map, page_no)
        next_page = _read_page(page_map, page_no + 1)

        per_page_chunks = _chunk_single_page_with_llm(
            target_page=target_page,
            next_page=next_page,
            page_images_dir=page_images_dir,
        )

        page_json_path = page_json_dir / f"page_{page_no:04d}.json"
        _save_page_json(page_json_path, per_page_chunks, fallback_page_no=page_no)

    return True