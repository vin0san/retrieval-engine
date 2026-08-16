import json
import logging
from pathlib import Path
import numpy as np
import torch
from numpy.linalg import norm
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "BAAI/bge-small-en-v1.5"
ONNX_OUT_DIR = Path("models/onnx/bge-small-onnx")
CHUNKS_PATH = Path("data/processed/chunks.json")


def export_to_onnx():
    """Use Optimum's ORTModelForFeatureExtraction.from_pretrained(..., export=True)
    to export + save both model and tokenizer to ONNX_OUT_DIR.
    """
    logger.info(f"Exporting PyTorch model '{MODEL_ID}' to ONNX format at '{ONNX_OUT_DIR}'...")
    ONNX_OUT_DIR.mkdir(parents=True, exist_ok=True)

    ort_model = ORTModelForFeatureExtraction.from_pretrained(MODEL_ID, export=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    ort_model.save_pretrained(ONNX_OUT_DIR)
    tokenizer.save_pretrained(ONNX_OUT_DIR)
    logger.info("ONNX export and tokenizer saved successfully!")


def load_pytorch_model():
    """AutoModel.from_pretrained(MODEL_ID), AutoTokenizer.from_pretrained(MODEL_ID)"""
    logger.info(f"Loading PyTorch model and tokenizer for '{MODEL_ID}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()
    return model, tokenizer


def load_onnx_model():
    """ORTModelForFeatureExtraction.from_pretrained(ONNX_OUT_DIR)"""
    logger.info(f"Loading ONNX model and tokenizer from '{ONNX_OUT_DIR}'...")
    ort_model = ORTModelForFeatureExtraction.from_pretrained(ONNX_OUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(ONNX_OUT_DIR)
    return ort_model, tokenizer


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling over token embeddings, masked by attention_mask.
    Also applies L2 normalization so cosine similarity reduces to inner product.
    """
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    )
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    pooled = sum_embeddings / sum_mask
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


def embed_pytorch(texts: list[str], model, tokenizer) -> np.ndarray:
    """Generate normalized embeddings via PyTorch runtime."""
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
    return pooled.cpu().numpy()


def embed_onnx(texts: list[str], ort_model, tokenizer) -> np.ndarray:
    """Generate normalized embeddings via Optimum ONNX Runtime."""
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    outputs = ort_model(**inputs)
    pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
    return pooled.cpu().numpy()


def validate(pt_embeddings: np.ndarray, onnx_embeddings: np.ndarray):
    """Cosine similarity per sample + max absolute difference evaluation."""
    logger.info("--- VALIDATION METRICS ---")

    max_abs_diff = np.max(np.abs(pt_embeddings - onnx_embeddings))
    logger.info(f"Max Absolute Difference: {max_abs_diff:.6e}")

    for idx, (pt_vec, onnx_vec) in enumerate(zip(pt_embeddings, onnx_embeddings)):
        cos_sim = np.dot(pt_vec, onnx_vec) / (norm(pt_vec) * norm(onnx_vec))
        logger.info(
            f"Sample {idx+1} | Cosine Similarity: {cos_sim:.8f} | Max Abs Diff: {np.max(np.abs(pt_vec - onnx_vec)):.6e}"
        )


def get_real_sample_chunks(num_samples: int = 5) -> list[str]:
    """Pull real chunk texts directly from data/processed/chunks.json."""
    if not CHUNKS_PATH.exists():
        logger.warning(f"'{CHUNKS_PATH}' not found. Falling back to default test strings.")
        return [
            "Optimizing transformer inference using ONNX Runtime dynamic batching.",
            "PagedAttention manages key-value cache memory using virtual memory page tables.",
            "CUDA Execution Provider enables high-performance GPU acceleration.",
            "Hierarchical Navigable Small World graphs provide microsecond vector retrieval.",
            "Quantization reduces memory footprint from float32 to 8-bit integers.",
        ]

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return [c["text"] for c in chunks[:num_samples]]


if __name__ == "__main__":
    export_to_onnx()

    sample_texts = get_real_sample_chunks(num_samples=5)

    pt_model, pt_tokenizer = load_pytorch_model()
    pt_embeddings = embed_pytorch(sample_texts, pt_model, pt_tokenizer)

    ort_model, ort_tokenizer = load_onnx_model()
    onnx_embeddings = embed_onnx(sample_texts, ort_model, ort_tokenizer)

    validate(pt_embeddings, onnx_embeddings)