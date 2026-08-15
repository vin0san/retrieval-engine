import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ingestion.loader import RawDoc

logger = logging.getLogger(__name__)

FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)


@dataclass
class Chunk:
    source: str
    relative_path: str
    heading: str
    chunk_index: int
    text: str
    token_count: int


def strip_noise(md_text: str) -> str:
    """Remove front-matter and trim leading/trailing whitespace."""
    if not md_text:
        return ""
    cleaned = FRONTMATTER_RE.sub("", md_text)
    return cleaned.strip()


def split_into_sections(md_text: str) -> list[tuple[str, str]]:
    """Split on headers -> [(heading, section_text), ...]"""
    sections: list[tuple[str, str]] = []

    if not md_text:
        return []

    matches = list(HEADER_RE.finditer(md_text))

    if not matches:
        return [("Overview", md_text.strip())]

    if matches[0].start() > 0:
        pre_text = md_text[: matches[0].start()].strip()
        if pre_text:
            sections.append(("Overview", pre_text))

    for i in range(len(matches)):
        heading = matches[i].group(2).strip()
        start_pos = matches[i].end()

        if i == len(matches) - 1:
            end_pos = len(md_text)
        else:
            end_pos = matches[i + 1].start()

        section_text = md_text[start_pos:end_pos].strip()

        if section_text:
            sections.append((heading, section_text))

    return sections


def _split_paragraphs_fence_aware(text: str) -> list[str]:
    """Split text on double-newlines (\\n\\n), ignoring blank lines inside code blocks."""
    if not text.strip():
        return []

    fence_spans = [(m.start(), m.end()) for m in CODE_BLOCK_RE.finditer(text)]

    def is_inside_fence(pos: int) -> bool:
        return any(start <= pos < end for start, end in fence_spans)

    paragraphs = []
    current_start = 0

    for match in re.finditer(r"\n\n+", text):
        split_pos = match.start()
        if not is_inside_fence(split_pos):
            para = text[current_start:split_pos].strip()
            if para:
                paragraphs.append(para)
            current_start = match.end()

    tail = text[current_start:].strip()
    if tail:
        paragraphs.append(tail)

    return paragraphs


def _finalize_chunk(text: str, tokenizer, max_tokens: int) -> str:
    """Re-encode assembled string and truncate if subword boundary merging exceeded budget."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text

    logger.warning(
        f"Chunk boundary merge exceeded budget ({len(ids)} > {max_tokens}). Hard truncating."
    )
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True).strip()


def _sliding_window_fallback(
    text: str, heading: str, tokenizer, max_tokens: int, overlap: int
) -> list[str]:
    """Fallback sliding window for single paragraphs that exceed max_tokens."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    header_prefix = f"[{heading}]\n"
    prefix_tokens = len(tokenizer.encode(header_prefix, add_special_tokens=False))

    effective_max = max(max_tokens - prefix_tokens - 6, 50)
    step = max(effective_max - overlap, 1)

    sub_chunks = []
    for start in range(0, len(token_ids), step):
        chunk_tokens = token_ids[start : start + effective_max]
        body = tokenizer.decode(chunk_tokens, skip_special_tokens=True).strip()
        if body:
            raw_assembled = f"{header_prefix}{body}"
            finalized = _finalize_chunk(raw_assembled, tokenizer, max_tokens)
            sub_chunks.append(finalized)

    return sub_chunks


def chunk_section(
    text: str,
    heading: str,
    tokenizer,
    max_tokens: int = 256,
    overlap: int = 40,
    min_tokens: int = 15,
) -> list[str]:
    """Fence-aware paragraph packing with sliding window fallback and hard boundary checking."""
    if not text.strip():
        return []

    header_prefix = f"[{heading}]\n"
    prefix_tokens = len(tokenizer.encode(header_prefix, add_special_tokens=False))

    effective_max = max(max_tokens - prefix_tokens - 6, 50)

    # fence-aware splitter
    paragraphs = _split_paragraphs_fence_aware(text)
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_paras: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(tokenizer.encode(para, add_special_tokens=False))

        # Single paragraph (or single entire code block) exceeds budget on its own
        if para_tokens > effective_max:
            if current_paras:
                chunk_body = "\n\n".join(current_paras)
                raw_chunk = f"{header_prefix}{chunk_body}"
                chunks.append(_finalize_chunk(raw_chunk, tokenizer, max_tokens))
                current_paras = []
                current_tokens = 0

            sub_chunks = _sliding_window_fallback(para, heading, tokenizer, max_tokens, overlap)
            chunks.extend(sub_chunks)
            continue

        if current_tokens + para_tokens > effective_max and current_paras:
            chunk_body = "\n\n".join(current_paras)
            raw_chunk = f"{header_prefix}{chunk_body}"
            chunks.append(_finalize_chunk(raw_chunk, tokenizer, max_tokens))

            carried_para = current_paras[-1]
            carried_tokens = len(tokenizer.encode(carried_para, add_special_tokens=False))

            if carried_tokens + para_tokens > effective_max:
                current_paras = []
                current_tokens = 0
            else:
                current_paras = [carried_para]
                current_tokens = carried_tokens

        current_paras.append(para)
        current_tokens += para_tokens

    if current_paras:
        chunk_body = "\n\n".join(current_paras)
        raw_chunk = f"{header_prefix}{chunk_body}"
        chunks.append(_finalize_chunk(raw_chunk, tokenizer, max_tokens))

    valid_chunks = [
        c for c in chunks 
        if len(tokenizer.encode(c, add_special_tokens=False)) >= min_tokens
    ]

    return valid_chunks


def chunk_doc(doc: "RawDoc", tokenizer) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0

    clean_text = strip_noise(doc.raw_text)
    sections = split_into_sections(clean_text)

    for header, content in sections:
        text_chunks = chunk_section(content, header, tokenizer)

        for text in text_chunks:
            token_count = len(tokenizer.encode(text, add_special_tokens=False))

            chunks.append(
                Chunk(
                    source=doc.source,
                    relative_path=doc.relative_path,
                    heading=header,
                    chunk_index=chunk_index,
                    text=text,
                    token_count=token_count,
                )
            )

            chunk_index += 1

    return chunks