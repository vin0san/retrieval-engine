import json
import logging
from dataclasses import asdict
from pathlib import Path
from transformers import AutoTokenizer

from src.ingestion.loader import load_all_sources
from src.ingestion.chunker import chunk_doc

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_ingestion():
    raw_root = Path("data/raw")
    output_path = Path("data/processed/chunks.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sources = ["vllm", "onnxruntime"]

    logger.info("Initializing Hugging Face tokenizer (BAAI/bge-small-en-v1.5)...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

    logger.info(f"Loading raw Markdown files from sources: {sources}")
    raw_docs = load_all_sources(raw_root, sources)
    logger.info(f"Loaded {len(raw_docs)} total raw documents.")

    all_chunks = []
    logger.info("Chunking documents using paragraph-aware windowing...")

    for doc in raw_docs:
        doc_chunks = chunk_doc(doc, tokenizer)
        all_chunks.extend(doc_chunks)

    logger.info(f"Generated {len(all_chunks)} total text chunks.")

    # Invariant & Distribution Check
    token_counts = [c.token_count for c in all_chunks]
    max_count = max(token_counts)
    min_count = min(token_counts)
    avg_count = sum(token_counts) / len(token_counts)

    logger.info("--- CHUNK DISTRIBUTION METRICS ---")
    logger.info(f"Total Chunks : {len(all_chunks)}")
    logger.info(f"Min Tokens   : {min_count}")
    logger.info(f"Max Tokens   : {max_count} (Budget Ceiling: 256)")
    logger.info(f"Avg Tokens   : {avg_count:.2f}")

    assert max_count <= 256, f"Invariant Violation: Chunk found with {max_count} tokens > 256"

    # Serialize to disk
    logger.info(f"Saving serialized chunks to {output_path}...")
    serializable_chunks = [asdict(c) for c in all_chunks]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_chunks, f, indent=2, ensure_ascii=False)

    logger.info("Ingestion pipeline complete!")


if __name__ == "__main__":
    run_ingestion()