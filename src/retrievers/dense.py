import logging
import os
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Literal, Tuple
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from config import (
    EMBEDDING_MODEL_NAME, EMBEDDINGS_PATH, DOC_IDS_PATH,
    DENSE_BATCH_SIZE
)

logger = logging.getLogger(__name__)

class DenseRetriever:
    def __init__(self, corpus: pd.DataFrame):
        """
        Initialize DenseRetriever.
        
        Args:
            corpus: DataFrame containing 'doc_id', 'text', 'title'.
        """
        self.corpus = corpus
        self.doc_ids = corpus['doc_id'].tolist()
        
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
            
        logger.info(f"Loading SentenceTransformer on device: {device}")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        
        self.embeddings = self._load_or_generate_embeddings()
        self.dimension = self.embeddings.shape[1]

    def _load_or_generate_embeddings(self) -> np.ndarray:
        """
        Loads embeddings from disk or generates them if not found.
        """
        if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(DOC_IDS_PATH):
            logger.info("Loading embeddings from cache...")
            embeddings = np.load(EMBEDDINGS_PATH)
            with open(DOC_IDS_PATH, 'rb') as f:
                cached_doc_ids = pickle.load(f)
            
            if cached_doc_ids == self.doc_ids:
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
            normalize_embeddings=True
        )
        
        # Save to cache
        np.save(EMBEDDINGS_PATH, embeddings)
        with open(DOC_IDS_PATH, 'wb') as f:
            pickle.dump(self.doc_ids, f)
            
        return embeddings


    def search(self, query: str, top_k: int = 1000) -> List[Tuple[str, float]]:
        """
        Search using Exact Search (Dot Product) via SentenceTransformers util.
        Annoy was found to be unreliable in this environment.
        
        Args:
            query: Query string.
            top_k: Number of results.
            
        Returns:
            List of tuples (doc_id, score).
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        
        # Convert corpus embeddings to tensor if they are numpy
        if isinstance(self.embeddings, np.ndarray):
            corpus_embeddings = torch.from_numpy(self.embeddings).to(query_embedding.device)
        else:
            corpus_embeddings = self.embeddings

        # Perform semantic search
        # returns list of list of dicts: [[{'corpus_id': int, 'score': float}, ...]]
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]
        
        results = []
        for hit in hits:
            doc_id = self.doc_ids[hit['corpus_id']]
            results.append((doc_id, float(hit['score'])))
            
        return results

