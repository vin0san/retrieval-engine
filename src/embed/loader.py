from huggingface_hub import snapshot_download
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer


def load_embedding_model(repo_id="vin0san/bge-small-onnx-quantized", subfolder="quantized"):
    local_dir = snapshot_download(repo_id=repo_id)
    ort_model = ORTModelForFeatureExtraction.from_pretrained(local_dir, subfolder=subfolder)
    tokenizer = AutoTokenizer.from_pretrained(local_dir, subfolder=subfolder)
    return ort_model, tokenizer