import logging
from typing import Dict
from config import RRF_K

logger = logging.getLogger(__name__)

def perform_rrf(
    sparse_results: Dict[str, float],
    dense_results: Dict[str, float],
    k: int = RRF_K
) -> Dict[str, float]:
    rrf_scores = {}
    
    def process_results(results: Dict[str, float]):
        sorted_docs = sorted(results.keys(), key=lambda x: results[x], reverse=True)
        for rank, doc_id in enumerate(sorted_docs, start=1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank)

    process_results(sparse_results)
    process_results(dense_results)
    
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return dict(sorted_rrf)

