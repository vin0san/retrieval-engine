import json
import logging
import time
from pathlib import Path
import numpy as np
import torch
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from src.embed.loader import load_embedding_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# optional
# QUANTIZED_MODEL_DIR = Path("models/onnx/bge-small-onnx-quantized")

CHUNKS_PATH = Path("data/processed/chunks.json")
OUTPUT_PATH = Path("data/processed/embeddings.npy")
BATCH_SIZE = 32


def load_chunk_texts(path: Path) -> list[str]:
    """Load chunks.json, return texts in exact file order.
    Order MUST match the output array row index i.
    """
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found at {path}. Run ingestion first.")

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]
    logger.info(f"Loaded {len(texts)} chunks from {path}")
    return texts


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mask-aware mean pooling over token embeddings.
    Applies L2 normalization so vector inner product equals cosine similarity.
    """
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    )
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    pooled = sum_embeddings / sum_mask
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


def embed_batch(texts: list[str], ort_model: ORTModelForFeatureExtraction, tokenizer: AutoTokenizer) -> np.ndarray:
    """Generate normalized embeddings for a batch of text strings."""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    outputs = ort_model(**inputs)
    pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
    return pooled.cpu().numpy()


def run_batched_embedding():
    """Execute batched inference over all ingested document chunks."""
    logger.info(f"Loading INT8 ONNX model from huggingface repo. ...")
    ort_model, tokenizer = load_embedding_model()

    texts = load_chunk_texts(CHUNKS_PATH)
    total_chunks = len(texts)

    all_embeddings = []
    batch_times = []

    logger.info(f"Starting batched embedding inference (Batch Size: {BATCH_SIZE})...")
    start_time = time.perf_counter()

    for i in range(0, total_chunks, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_start = time.perf_counter()

        emb = embed_batch(batch, ort_model, tokenizer)

        batch_duration = time.perf_counter() - batch_start
        batch_times.append(batch_duration)
        all_embeddings.append(emb)

        if (len(batch_times) % 10 == 0) or (i + BATCH_SIZE >= total_chunks):
            processed = min(i + BATCH_SIZE, total_chunks)
            logger.info(
                f"Processed {processed}/{total_chunks} chunks "
                f"({(processed / total_chunks) * 100:.1f}%) | "
                f"Last batch time: {batch_duration * 1000:.2f} ms"
            )

    total_time = time.perf_counter() - start_time
    embeddings = np.vstack(all_embeddings)

    # -------------------------------------------------------------------
    # Profiling & Metrics Logging
    # -------------------------------------------------------------------
    throughput = total_chunks / total_time
    avg_batch_ms = np.mean(batch_times) * 1000
    min_batch_ms = np.min(batch_times) * 1000
    max_batch_ms = np.max(batch_times) * 1000

    logger.info("==================================================")
    logger.info("           INFERENCE PROFILING DATA               ")
    logger.info("==================================================")
    logger.info(f"Total Time      : {total_time:.2f} s")
    logger.info(f"Throughput      : {throughput:.2f} chunks/sec")
    logger.info(f"Avg Batch Time  : {avg_batch_ms:.2f} ms")
    logger.info(f"Min Batch Time  : {min_batch_ms:.2f} ms")
    logger.info(f"Max Batch Time  : {max_batch_ms:.2f} ms")
    logger.info("==================================================")

    # -------------------------------------------------------------------
    # Invariant & Sanity Checks
    # -------------------------------------------------------------------
    logger.info("Running output sanity checks...")

    # 1. Shape Check
    expected_shape = (total_chunks, 384)
    assert embeddings.shape == expected_shape, (
        f"Shape mismatch! Expected {expected_shape}, got {embeddings.shape}"
    )
    logger.info(f"✓ Shape check passed: {embeddings.shape}")

    # 2. NaN / Inf Check
    assert not np.isnan(embeddings).any(), "Sanity Check Failed: NaN values found in embeddings array!"
    assert not np.isinf(embeddings).any(), "Sanity Check Failed: Inf values found in embeddings array!"
    logger.info("✓ NaN/Inf check passed.")

    # 3. L2 Norm Invariant Check (Norms should be ~1.0 for every row)
    norms = np.linalg.norm(embeddings, axis=1)
    norm_diffs = np.abs(norms - 1.0)
    max_norm_diff = np.max(norm_diffs)
    assert max_norm_diff < 1e-3, (
        f"L2 Normalization Invariant Violation! Max deviation from 1.0 is {max_norm_diff:.6f}"
    )
    logger.info(f"✓ L2 Norm check passed: All rows unit normalized (Max deviation from 1.0: {max_norm_diff:.6e})")

    # Save to disk
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, embeddings)
    logger.info(f"Successfully saved embeddings to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_batched_embedding()