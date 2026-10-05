import numpy as np

def brute_force_search(query_embedding: np.ndarray, corpus_embeddings: np.ndarray, k: int = 5) -> np.ndarray:
    """Cosine similarity search via dot product (embeddings are pre-normalized).
    Returns indices of the top-k nearest neighbors, most similar first."""
    similarities = corpus_embeddings @ query_embedding
    top_k_indices = np.argsort(-similarities)[:k]
    return top_k_indices