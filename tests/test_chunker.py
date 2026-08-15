import pytest
from transformers import AutoTokenizer

from src.ingestion.chunker import (
    Chunk,
    _finalize_chunk,
    _split_paragraphs_fence_aware,
    _sliding_window_fallback,
    chunk_doc,
    chunk_section,
    split_into_sections,
    strip_noise,
)
from src.ingestion.loader import RawDoc


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")


class TestStripNoise:
    def test_strip_frontmatter(self):
        raw_md = "---\ntitle: Test Doc\nauthor: Vineet\n---\n\n# Real Content\nThis is a test."
        cleaned = strip_noise(raw_md)
        assert "title: Test Doc" not in cleaned
        assert "author: Vineet" not in cleaned
        assert cleaned == "# Real Content\nThis is a test."

    def test_strip_empty_string(self):
        assert strip_noise("") == ""
        assert strip_noise(None) == ""


class TestSplitIntoSections:
    def test_no_headers_fallback_to_overview(self):
        text = "This is a plain document without any markdown headers."
        sections = split_into_sections(text)
        assert len(sections) == 1
        assert sections[0] == ("Overview", text)

    def test_pre_header_overview_and_sections(self):
        text = """Pre-header introductory paragraph.

# Section One
Body content of section 1.

## Section Two
Body content of section 2.
"""
        sections = split_into_sections(text)
        assert len(sections) == 3
        assert sections[0] == ("Overview", "Pre-header introductory paragraph.")
        assert sections[1] == ("Section One", "Body content of section 1.")
        assert sections[2] == ("Section Two", "Body content of section 2.")

    def test_empty_input(self):
        assert split_into_sections("") == []


class TestFenceAwareParagraphSplitter:
    def test_splits_paragraphs_outside_code_blocks(self):
        text = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        paras = _split_paragraphs_fence_aware(text)
        assert paras == ["Paragraph 1", "Paragraph 2", "Paragraph 3"]

    def test_preserves_blank_lines_inside_code_fences(self):
        text = """First paragraph.

```python
def process():
    # Step 1

    # Step 2
    return True
```

Last paragraph."""
        paras = _split_paragraphs_fence_aware(text)
        assert len(paras) == 3
        assert paras[0] == "First paragraph."
        assert "def process():" in paras[1]
        assert "# Step 2" in paras[1]
        assert paras[2] == "Last paragraph."


class TestFinalizeChunk:
    def test_within_budget_unmodified(self, tokenizer):
        text = "[Overview]\nShort text under budget."
        finalized = _finalize_chunk(text, tokenizer, max_tokens=256)
        assert finalized == text

    def test_exceeds_budget_truncated(self, tokenizer):
        long_text = "word " * 500
        max_tokens = 50
        finalized = _finalize_chunk(long_text, tokenizer, max_tokens=max_tokens)
        tokens = tokenizer.encode(finalized, add_special_tokens=False)
        assert len(tokens) <= max_tokens


class TestSlidingWindowFallback:
    def test_large_single_paragraph_chunking(self, tokenizer):
        large_para = "Inference optimization requires efficient memory management and kernel fusion. " * 30
        heading = "Optimization Strategy"
        max_tokens = 100
        overlap = 20

        sub_chunks = _sliding_window_fallback(large_para, heading, tokenizer, max_tokens, overlap)
        assert len(sub_chunks) > 1

        for sc in sub_chunks:
            assert sc.startswith(f"[{heading}]\n")
            tokens = tokenizer.encode(sc, add_special_tokens=False)
            assert len(tokens) <= max_tokens


class TestChunkSectionAndDoc:
    def test_chunk_section_token_constraints(self, tokenizer):
        text = "Deep learning compiler optimizations transform high-level computational graphs. " * 15
        chunks = chunk_section(
            text=text,
            heading="Compilers",
            tokenizer=tokenizer,
            max_tokens=128,
            min_tokens=15,
        )

        assert len(chunks) >= 1
        for c in chunks:
            tokens = len(tokenizer.encode(c, add_special_tokens=False))
            assert 15 <= tokens <= 128
            assert c.startswith("[Compilers]\n")

    def test_chunk_doc_end_to_end(self, tokenizer):
        doc = RawDoc(
            source="vllm",
            relative_path="engine/runner.md",
            raw_text="""---
title: Engine Execution
---

# PagedAttention Kernel
PagedAttention manages key-value cache memory using virtual memory page tables to eliminate external fragmentation.

```python
def allocate_blocks(num_tokens):
    return page_table.allocate(num_tokens)
```

## Performance Metrics
Latency and throughput metrics demonstrate linear scaling with sequence length up to maximum context windows.
""",
        )

        chunks = chunk_doc(doc, tokenizer)
        assert len(chunks) >= 2

        for idx, chunk in enumerate(chunks):
            assert isinstance(chunk, Chunk)
            assert chunk.source == "vllm"
            assert chunk.relative_path == "engine/runner.md"
            assert chunk.chunk_index == idx
            assert chunk.token_count == len(tokenizer.encode(chunk.text, add_special_tokens=False))
            assert chunk.token_count >= 15
            assert chunk.token_count <= 256
