from pathlib import Path
import pytest
from src.ingestion.loader import RawDoc, load_all_sources, load_source


def test_load_source_success(tmp_path: Path):
    """Verify loading valid markdown files recursively while skipping non-md/empty files."""
    source_dir = tmp_path / "vllm"
    source_dir.mkdir()

    sub_dir = source_dir / "docs"
    sub_dir.mkdir()

    # Valid files
    (source_dir / "index.md").write_text("# Welcome\nIndex page", encoding="utf-8")
    (sub_dir / "setup.md").write_text("# Setup\nInstallation guide", encoding="utf-8")

    # Edge cases to skip
    (source_dir / "empty.md").write_text("   \n  ", encoding="utf-8")
    (source_dir / "config.json").write_text('{"key": "value"}', encoding="utf-8")

    docs = load_source(source_dir, "vllm")

    assert len(docs) == 2
    paths = {d.relative_path for d in docs}
    assert "index.md" in paths
    assert str(Path("docs") / "setup.md") in paths

    for doc in docs:
        assert isinstance(doc, RawDoc)
        assert doc.source == "vllm"
        assert len(doc.raw_text.strip()) > 0


def test_load_all_sources(tmp_path: Path):
    """Verify multi-source loading and missing directory resilience."""
    raw_root = tmp_path / "raw"
    raw_root.mkdir()

    vllm_dir = raw_root / "vllm"
    vllm_dir.mkdir()
    (vllm_dir / "doc1.md").write_text("vllm doc content", encoding="utf-8")

    onnx_dir = raw_root / "onnxruntime"
    onnx_dir.mkdir()
    (onnx_dir / "doc2.md").write_text("onnx doc content", encoding="utf-8")

    # Passing a non-existent source name shouldn't throw an exception
    docs = load_all_sources(raw_root, ["vllm", "onnxruntime", "missing_source"])

    assert len(docs) == 2
    sources = {d.source for d in docs}
    assert sources == {"vllm", "onnxruntime"}