import logging
from typing import Dict, List
import pandas as pd
import numpy as np
from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL_NAME, TOP_K_RERANK

logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, corpus: pd.DataFrame):
        """
        Initialize Reranker.
        
        Args:
            corpus: DataFrame containing 'doc_id', 'text', 'title'.
        """
        self.corpus = corpus.set_index('doc_id')
        logger.info(f"Initializing Cross-Encoder model: {RERANKER_MODEL_NAME}")
        self.model = CrossEncoder(RERANKER_MODEL_NAME)

    def rerank(self, query: str, initial_results: Dict[str, float]) -> Dict[str, float]:
        """
        Reranks the top K retrieved documents using a Cross-Encoder model.
        
        Args:
            query: The search query.
            initial_results: Dictionary mapping doc_ids to their initial retrieval scores.
            
        Returns:
            Dictionary of {doc_id: new_score}, containing both reranked top-K docs 
            and the remaining docs with adjusted scores to maintain relative ordering.
        """
        # Sort results by score descending
        sorted_docs = sorted(initial_results.items(), key=lambda x: x[1], reverse=True)
        
        # Slice top-k for reranking, keep the rest as-is (for now)
        top_k_docs = sorted_docs[:TOP_K_RERANK]
        rest_docs = sorted_docs[TOP_K_RERANK:]
        
        if not top_k_docs:
            return {}
            
        # Prepare input pairs for the model
        pairs = []
        doc_ids_to_rerank = []
        
        for doc_id, _ in top_k_docs:
            if doc_id in self.corpus.index:
                text = self.corpus.loc[doc_id]['text']
                title = self.corpus.loc[doc_id]['title']
                # Concatenate title and text for better context
                content = f"{title} {text}"
                pairs.append([query, content])
                doc_ids_to_rerank.append(doc_id)
            else:
                logger.warning(f"Skipping rerank for missing doc_id: {doc_id}")
        
        if not pairs:
            return dict(sorted_docs)

        # Run inference
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Construct the final results dictionary
        reranked_results = {}
        min_reranked_score = float('inf')
        
        for doc_id, score in zip(doc_ids_to_rerank, scores):
            reranked_results[doc_id] = float(score)
            if score < min_reranked_score:
                min_reranked_score = score
                
        # Handle the tail: shift scores of non-reranked docs to sit below the lowest reranked score.
        # This ensures the reranked set stays at the top while preserving the original order of the rest.
        # Formula: new_score = min_reranked - 1.0 - (rank_offset * small_epsilon)
        
        for i, (doc_id, _) in enumerate(rest_docs):
            reranked_results[doc_id] = min_reranked_score - 1.0 - (i * 0.001)
            
        return reranked_results

