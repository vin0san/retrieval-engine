# rag-hnsw-system

A retrieval-augmented QA system over technical documentation, built to demonstrate 
inference-serving and ANN systems engineering — not framework-wrapper fluency.

## Why This Project

Most RAG portfolios wire together LangChain and a vector DB, which proves API 
familiarity, not systems understanding. This project's implements HNSW from first principles to study graph-based ANN indexing and compare it against established libraries. used as the actual retrieval index, an 
ONNX-quantized embedding pipeline, and cross-pipeline latency profiling — the same 
inference-optimization focus behind my other projects (skin-lesion-onnx-api).

## What's Raw vs. What's a Library

Deliberate, not default. Solved infrastructure (serialization, HTTP framework, 
LLM API calls) is used as-is. Anything that's the actual skill being demonstrated 
(chunking, HNSW, batched inference serving, profiling) is hand-built.

| Layer | Raw | Library |
|---|---|---|
| Ingestion | Markdown loading, chunking | jsonl/parquet storage |
| Embedding | Batched inference serving | ONNX export, quantization (`optimum`, `onnxruntime`) |
| Retrieval core | HNSW, brute-force ground truth, profiling | FAISS (v2 baseline comparison only) |
| Generation | — | Thin LLM API wrapper (deliberately minimal — not this project's focus) |
| Serving | — | FastAPI, Docker (reusing prior project's competency) |

## Corpus

Current corpus:
- ONNX Runtime documentation
- vLLM documentation

(Current chunk count depends on ingestion configuration.)

chosen specifically to avoid AST-parsing/PDF-extraction detours 
that don't serve the core differentiator. Scale for the actual HNSW-vs-FAISS 
benchmark comes from a public ANN dataset in v2, not the doc corpus — brute-force 
search over a few thousand vectors is sub-millisecond regardless of index quality, 
so a meaningful latency comparison needs real scale, which doc scraping can't provide.

## Architecture

```text
Raw Markdown Documents
        │
        ▼
     Loader
        │
        ▼
      RawDoc
        │
        ▼
     Chunker
        │
        ▼
      Chunk[]
        │
        ▼
   chunks.json
```

## Current Status

**Phase 1 (Ingestion) — complete.**

- Recursive markdown loading, tagged by source
- Frontmatter removal, code-fence-aware paragraph splitting
- Token-aware chunking (BAAI/bge-small-en-v1.5 tokenizer) with sliding-window 
  fallback for oversized paragraphs, hard budget verification
- 7,083 chunks generated and validated (before final corpus-level checks)

## Roadmap

### v1 — Working, Correctness-Validated Pipeline

- [x] Markdown ingestion + chunking
- [ ] Optimize embedding inference through ONNX Runtime, 
      then investigate quantization and measure its impact on throughput and latency, batched serving
- [ ] Custom HNSW implementation
- [ ] Recall@k validation against brute-force ground truth (correctness proof, 
      not a formal eval harness)
- [ ] Cross-pipeline profiling (embed / search / generation time) — the 
      connective narrative tying the whole pipeline together
- [ ] Thin generation wrapper
- [ ] FastAPI serving layer, containerized, deployed demo

### v2 — Stress Test (deferred, explicit reasons below)

- [ ] Large-scale HNSW vs. FAISS benchmark on a public ANN dataset (SIFT1M or 
      GloVe) — recall/latency/build-time curves at real scale
- [ ] C++ port of the HNSW hot path

### Considered, Deliberately Deferred

These are real, valuable skills that belong to a different project's story 
(search-relevance engineering), and stacking them here would dilute one sharp 
narrative into several shallow ones:

- Hybrid retrieval (BM25 + dense fusion), reranking
- Formal retrieval-quality eval harness (MRR, Precision@k) — the recall@k check 
  above is a correctness proof, not this
- AST-aware chunking
- Swappable vector-store backends (pgvector, etc.) — no driving need identified

## Stack

Python, PyTorch → ONNX, `onnxruntime`, FastAPI, Docker.