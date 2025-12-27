from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()

CACHE_DIR = PROJECT_ROOT / "cache"
EMBEDDINGS_PATH = CACHE_DIR / "corpus_embeddings.npy"
DOC_IDS_PATH = CACHE_DIR / "doc_ids.pkl"
HNSW_INDEX_PATH = CACHE_DIR / "hnsw_index.bin"

CACHE_DIR.mkdir(exist_ok=True)

DATASET_NAME = "beir/trec-covid"

EMBEDDING_MODEL_NAME = "all-MiniLM-L12-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

BM25_K1 = 1.2
BM25_B = 0.75

DENSE_BATCH_SIZE = 512
HNSW_M = 20
HNSW_EF_CONSTRUCTION = 200
HNSW_SPACE = "cosine"

RRF_K = 60

TOP_K_RERANK = 50

