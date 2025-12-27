import os
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data & Cache Paths
CACHE_DIR = PROJECT_ROOT / "cache"
EMBEDDINGS_PATH = CACHE_DIR / "corpus_embeddings.npy"
DOC_IDS_PATH = CACHE_DIR / "doc_ids.pkl"

# Ensure directories exist
CACHE_DIR.mkdir(exist_ok=True)

# Dataset
DATASET_NAME = "beir/trec-covid"

# Model Names
EMBEDDING_MODEL_NAME = "all-MiniLM-L12-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval Parameters
BM25_K1 = 1.2
BM25_B = 0.75

# Dense Retrieval Parameters
DENSE_BATCH_SIZE = 256

# Hybrid Parameters
RRF_K = 60

# Reranking Parameters
TOP_K_RERANK = 50

