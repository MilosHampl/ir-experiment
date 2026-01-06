import logging
import time
from typing import List, Dict
import pandas as pd
from rank_bm25 import BM25Okapi
from config import BM25_K1, BM25_B
from src.preprocessing import preprocess_text

logger = logging.getLogger(__name__)

class SparseRetriever:
    def __init__(self, corpus: pd.DataFrame, preprocess: bool = True):
        self.corpus = corpus
        self.doc_ids = corpus['doc_id'].tolist()
        self.preprocess = preprocess
        self.query_latencies: List[float] = []
        
        logger.info(f"Preprocessing corpus for BM25 (preprocess={preprocess})...")
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
        start_time = time.time()
        
        if self.preprocess:
            tokenized_query = preprocess_text(query)
        else:
            tokenized_query = query.split()
            
        scores = self.bm25.get_scores(tokenized_query)
        doc_scores = zip(self.doc_ids, scores)
        sorted_scores = sorted(doc_scores, key=lambda x: x[1], reverse=True)
        
        latency = time.time() - start_time
        self.query_latencies.append(latency)
        
        return {doc_id: score for doc_id, score in sorted_scores[:top_k]}
    
    def get_avg_query_latency(self) -> float:
        if not self.query_latencies:
            return 0.0
        return sum(self.query_latencies) / len(self.query_latencies)
