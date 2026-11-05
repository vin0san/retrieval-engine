import random
import time
import json
import logging
from pathlib import Path
import numpy as np

from src.index.hnsw import HNSW
from src.index.brute_force import brute_force_search
from src.embed.batch_embed import embed_query
from src.embed.loader import load_embedding_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
CHUNKS_PATH = Path("data/processed/chunks.json")

TEST_QUERIES = [
    "How does PagedAttention reduce memory fragmentation?",
    "ONNX Runtime CUDA execution provider optimization",
    "Quantization techniques for transformer models",
    "How to configure vLLM engine runner",
    "Text embedding mean pooling with attention mask",
]


def load_sample_queries(num_samples: int = 25, seed: int = 7) -> list[str]:
    """Use real corpus chunks as additional test queries."""
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    random.seed(seed)
    sampled = random.sample(chunks, num_samples)
    return [c["text"] for c in sampled]


def build_and_evaluate_hnsw() -> dict:
    logger.info(f"Loading corpus embeddings from {EMBEDDINGS_PATH}...")
    corpus = np.load(EMBEDDINGS_PATH)
    all_queries = TEST_QUERIES + load_sample_queries(num_samples=25)
    logger.info(f"Evaluating across {len(all_queries)} queries...")
    n_samples, dim = corpus.shape
    logger.info(f"Loaded matrix of shape {corpus.shape}")

    # Initialize HNSW index
    hnsw = HNSW(dim=dim, M=16, ef_construction=200, seed=42)

    logger.info("Building HNSW graph index...")
    build_start = time.perf_counter()
    for idx in range(n_samples):
        hnsw.insert(idx, corpus[idx])
        if (idx + 1) % 1000 == 0 or (idx + 1) == n_samples:
            logger.info(f"Inserted {idx + 1}/{n_samples} vectors into HNSW graph...")
    build_time = time.perf_counter() - build_start
    logger.info(f"✓ HNSW Graph Built in {build_time:.2f} seconds ({n_samples / build_time:.2f} insertions/sec)")

    # Load ONNX model
    logger.info("Loading INT8 ONNX model...")
    ort_model, tokenizer = load_embedding_model()

    K = 10
    EF_SEARCH = 50
    recalls, hnsw_latencies, bf_latencies = [], [], []

    logger.info("\nEvaluating HNSW vs Brute-Force Baseline...")
    for idx, q_text in enumerate(all_queries, 1):
        q_vec = embed_query(q_text, ort_model, tokenizer)

        # Brute force ground truth
        bf_start = time.perf_counter()
        gt_indices = brute_force_search(q_vec, corpus, k=K)
        bf_time = (time.perf_counter() - bf_start) * 1000
        bf_latencies.append(bf_time)

        # HNSW search
        hnsw_start = time.perf_counter()
        hnsw_indices = hnsw.search(q_vec, k=K, ef_search=EF_SEARCH)
        hnsw_time = (time.perf_counter() - hnsw_start) * 1000
        hnsw_latencies.append(hnsw_time)

        # Calculate Recall@K
        intersection = set(gt_indices).intersection(set(hnsw_indices))
        recall = len(intersection) / K
        recalls.append(recall)

        logger.info(
            f"[{idx}/{len(all_queries)}] Query: '{q_text[:40]}...' | "
            f"Recall@{K}: {recall * 100:.1f}% | "
            f"HNSW: {hnsw_time:.2f} ms | BF: {bf_time:.2f} ms"
        )

    avg_recall = np.mean(recalls) * 100
    avg_hnsw_ms = np.mean(hnsw_latencies)
    avg_bf_ms = np.mean(bf_latencies)

    logger.info("==================================================")
    logger.info("              BENCHMARK SUMMARY                   ")
    logger.info("==================================================")
    logger.info(f"Average Recall@{K}    : {avg_recall:.2f}%")
    logger.info(f"Avg HNSW Search Time : {avg_hnsw_ms:.3f} ms/query")
    logger.info(f"Avg Brute Force Time : {avg_bf_ms:.3f} ms/query")
    logger.info(f"Speedup Ratio        : {avg_bf_ms / avg_hnsw_ms:.2f}x")
    logger.info("==================================================")

    results = {
        "n_chunks": int(n_samples),
        "build_time_sec": float(build_time),
        "insertions_per_sec": float(n_samples / build_time),
        "avg_recall_at_10": float(avg_recall),
        "avg_hnsw_ms": float(avg_hnsw_ms),
        "avg_bf_ms": float(avg_bf_ms),
        "speedup_ratio": float(avg_bf_ms / avg_hnsw_ms),
        "n_queries": len(all_queries),
    }

    return results


if __name__ == "__main__":
    results = build_and_evaluate_hnsw()
    print("\nSerialized Benchmark Output:")
    print(results)