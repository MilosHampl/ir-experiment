import logging
from typing import List, Dict, Any
import pandas as pd
from rank_bm25 import BM25Okapi
from config import BM25_K1, BM25_B
from src.preprocessing import preprocess_text

logger = logging.getLogger(__name__)

class SparseRetriever:
    def __init__(self, corpus: pd.DataFrame, preprocess: bool = True):
        """
        Initialize SparseRetriever with BM25Okapi.
        
        Args:
            corpus: DataFrame containing 'doc_id', 'text', 'title'.
            preprocess: Whether to apply full preprocessing (True) or just whitespace tokenization (False).
        """
        self.corpus = corpus
        self.doc_ids = corpus['doc_id'].tolist()
        self.preprocess = preprocess
        
        logger.info(f"Preprocessing corpus for BM25 (preprocess={preprocess})...")
        # Combine title and text for better retrieval
        if self.preprocess:
            self.tokenized_corpus = [
                preprocess_text(f"{row['title']} {row['text']}") 
                for _, row in corpus.iterrows()
            ]
        else:
            self.tokenized_corpus = [
                f"{row['title']} {row['text']}".split()
                for _, row in corpus.iterrows()
            ]
        
        logger.info("Initializing BM25Okapi...")
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=BM25_K1,
            b=BM25_B
        )
        logger.info("BM25 initialized.")

    def search(self, query: str, top_k: int = 1000) -> Dict[str, float]:
        """
        Search the corpus using BM25.
        
        Args:
            query: Query string.
            top_k: Number of top results to return.
            
        Returns:
            Dictionary mapping doc_id to score.
        """
        if self.preprocess:
            tokenized_query = preprocess_text(query)
        else:
            tokenized_query = query.split()
            
        scores = self.bm25.get_scores(tokenized_query)
        
        # Pair scores with doc_ids
        doc_scores = zip(self.doc_ids, scores)
        
        # Sort by score descending
        sorted_scores = sorted(doc_scores, key=lambda x: x[1], reverse=True)
        
        # Return top_k as dict
        return {doc_id: score for doc_id, score in sorted_scores[:top_k]}
