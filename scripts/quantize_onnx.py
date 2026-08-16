import logging
import shutil
from pathlib import Path
import numpy as np
import torch
from numpy.linalg import norm
from onnxruntime.quantization import QuantType, quantize_dynamic
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

from scripts.export_onnx import embed_onnx, get_real_sample_chunks, load_onnx_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ONNX_OUT_DIR = Path("models/onnx/bge-small-onnx")
QUANTIZED_OUT_DIR = Path("models/onnx/bge-small-onnx-quantized")


def quantize_model(onnx_dir: Path, quantized_dir: Path):
    """Dynamic INT8 quantization of the exported ONNX FP32 model."""
    quantized_dir.mkdir(parents=True, exist_ok=True)

    model_input = onnx_dir / "model.onnx"
    model_output = quantized_dir / "model.onnx"

    if not model_input.exists():
        raise FileNotFoundError(f"Base ONNX model not found at '{model_input}'. Run export_onnx.py first.")

    logger.info(f"Quantizing ONNX FP32 model from '{model_input}' -> '{model_output}'...")
    quantize_dynamic(
        model_input=str(model_input),
        model_output=str(model_output),
        weight_type=QuantType.QUInt8,
    )

    # Copy non-model artifacts (tokenizer config, vocab, special tokens map, etc.)
    for file in onnx_dir.glob("*"):
        if file.name != "model.onnx":
            shutil.copy(file, quantized_dir / file.name)

    logger.info(f"Quantized model and tokenizer files saved to '{quantized_dir}'")


def load_quantized_model():
    """Load the quantized ONNX model and tokenizer using Optimum."""
    logger.info(f"Loading Quantized INT8 model from '{QUANTIZED_OUT_DIR}'...")
    ort_model = ORTModelForFeatureExtraction.from_pretrained(QUANTIZED_OUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(QUANTIZED_OUT_DIR)
    return ort_model, tokenizer


def validate_quantization(fp32_embeddings: np.ndarray, int8_embeddings: np.ndarray):
    """Evaluate FP32 vs INT8 embeddings (expected high cosine similarity ~0.97-0.995)."""
    logger.info("--- QUANTIZATION VALIDATION (FP32 vs INT8) ---")
    max_abs_diff = np.max(np.abs(fp32_embeddings - int8_embeddings))
    logger.info(f"Max Absolute Difference: {max_abs_diff:.6e}")

    for idx, (fp32_vec, int8_vec) in enumerate(zip(fp32_embeddings, int8_embeddings)):
        cos_sim = np.dot(fp32_vec, int8_vec) / (norm(fp32_vec) * norm(int8_vec))
        logger.info(f"Sample {idx+1} | FP32 vs INT8 Cosine Similarity: {cos_sim:.6f}")


if __name__ == "__main__":
    quantize_model(ONNX_OUT_DIR, QUANTIZED_OUT_DIR)

    sample_texts = get_real_sample_chunks(num_samples=5)

    # Run FP32 ONNX inference baseline
    ort_fp32_model, ort_fp32_tokenizer = load_onnx_model()
    fp32_embeddings = embed_onnx(sample_texts, ort_fp32_model, ort_fp32_tokenizer)

    # Run INT8 ONNX inference
    ort_int8_model, ort_int8_tokenizer = load_quantized_model()
    int8_embeddings = embed_onnx(sample_texts, ort_int8_model, ort_int8_tokenizer)

    validate_quantization(fp32_embeddings, int8_embeddings)