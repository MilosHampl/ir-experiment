import logging
import os
import pickle
import time
import numpy as np
import pandas as pd
import torch
from typing import Dict, List
from sentence_transformers import SentenceTransformer
import hnswlib
from config import (
    EMBEDDING_MODEL_NAME, EMBEDDINGS_PATH, DOC_IDS_PATH, HNSW_INDEX_PATH,
    DENSE_BATCH_SIZE, HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_SPACE
)

logger = logging.getLogger(__name__)

class DenseRetriever:
    def __init__(self, corpus: pd.DataFrame):
        self.corpus = corpus
        self.doc_ids = corpus['doc_id'].tolist()
        
        self.query_latencies: List[float] = []
        
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
            
        logger.info(f"Loading SentenceTransformer on device: {device}")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        
        self.embeddings = self._load_or_generate_embeddings()
        self.dimension = self.embeddings.shape[1]
        self.build_indices()

    def _load_or_generate_embeddings(self) -> np.ndarray:
        if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(DOC_IDS_PATH):
            logger.info("Loading embeddings from cache...")
            embeddings = np.load(EMBEDDINGS_PATH)
            with open(DOC_IDS_PATH, 'rb') as f:
                cached_doc_ids = pickle.load(f)
            
            if cached_doc_ids == self.doc_ids:
                logger.info("Embeddings loaded from cache.")
                return embeddings
            else:
                logger.warning("Cached doc_ids do not match current corpus. Regenerating...")
        
        logger.info("Generating embeddings...")
        texts = [f"{row['title']} {row['text']}" for _, row in self.corpus.iterrows()]
        
        embeddings = self.model.encode(
            texts,
            batch_size=DENSE_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.model.device
        )
        
        np.save(EMBEDDINGS_PATH, embeddings)
        with open(DOC_IDS_PATH, 'wb') as f:
            pickle.dump(self.doc_ids, f)
            
        return embeddings

    def build_indices(self):
        self.hnsw_index = hnswlib.Index(space=HNSW_SPACE, dim=self.dimension)

        if os.path.exists(HNSW_INDEX_PATH):
            logger.info("Loading HNSW index from cache...")
            try:
                self.hnsw_index.load_index(str(HNSW_INDEX_PATH))
                logger.info("HNSW index loaded.")
                return
            except Exception as e:
                logger.error(f"Failed to load HNSW index: {e}")

        logger.info("Building HNSW index...")
        self.hnsw_index.init_index(max_elements=len(self.embeddings), ef_construction=HNSW_EF_CONSTRUCTION, M=HNSW_M)
        self.hnsw_index.add_items(self.embeddings, np.arange(len(self.embeddings)))
            
        logger.info("HNSW index built. Saving to cache...")
        try:
            self.hnsw_index.save_index(str(HNSW_INDEX_PATH))
            logger.info("HNSW index saved.")
        except Exception as e:
            logger.error(f"Failed to save HNSW index: {e}")

    def search(self, query: str, top_k: int = 1000) -> Dict[str, float]:
        start_time = time.time()
        
        query_embedding = self.model.encode(query, convert_to_tensor=False, normalize_embeddings=True, show_progress_bar=False)
        labels, distances = self.hnsw_index.knn_query(query_embedding, k=top_k)
        
        results = {}
        for label, distance in zip(labels[0], distances[0]):
            doc_id = self.doc_ids[label]
            score = 1 - distance
            results[doc_id] = float(score)
        
        latency = time.time() - start_time
        self.query_latencies.append(latency)
            
        return results
    
    def get_avg_query_latency(self) -> float:
        if not self.query_latencies:
            return 0.0
        return sum(self.query_latencies) / len(self.query_latencies)

