import logging
from typing import Dict
from config import RRF_K

logger = logging.getLogger(__name__)

def perform_rrf(
    sparse_results: Dict[str, float],
    dense_results: Dict[str, float],
    k: int = RRF_K
) -> Dict[str, float]:
    """
    Perform Reciprocal Rank Fusion (RRF) on two sets of results.
    
    RRF Formula: Score = sum(1 / (k + rank_i))
    
    Args:
        sparse_results: Dictionary of {doc_id: score} from sparse retrieval.
        dense_results: Dictionary of {doc_id: score} from dense retrieval.
        k: Constant k for RRF (default 60).
        
    Returns:
        Dictionary of {doc_id: rrf_score}, sorted by score descending.
    """
    rrf_scores = {}
    
    # Helper to process a result set
    def process_results(results: Dict[str, float]):
        # Sort by score descending to determine rank
        sorted_docs = sorted(results.keys(), key=lambda x: results[x], reverse=True)
        
        for rank, doc_id in enumerate(sorted_docs, start=1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            
            # Add component score
            rrf_scores[doc_id] += 1.0 / (k + rank)

    # Process both result sets
    process_results(sparse_results)
    process_results(dense_results)
    
    # Sort final results by RRF score descending
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return dict(sorted_rrf)

