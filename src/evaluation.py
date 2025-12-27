import logging
import pandas as pd
import ir_measures
from ir_measures import nDCG, R, AP, P, RR
from typing import Dict, Any

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, qrels: pd.DataFrame):
        self.qrels = qrels
        self.metrics = [nDCG@10, R@100, AP, P@10, RR]

    def evaluate(self, run_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        logger.info("Converting run results to DataFrame...")
        run_data = []
        for qid, docs in run_results.items():
            for doc_id, score in docs.items():
                run_data.append({
                    "query_id": str(qid),
                    "doc_id": str(doc_id),
                    "score": score
                })
        
        if not run_data:
            logger.warning("No results to evaluate.")
            return {str(m): 0.0 for m in self.metrics}
            
        run_df = pd.DataFrame(run_data)
        
        logger.info("Calculating metrics...")
        qrels_df = self.qrels.copy()
        qrels_df['query_id'] = qrels_df['query_id'].astype(str)
        qrels_df['doc_id'] = qrels_df['doc_id'].astype(str)
        
        results = ir_measures.calc_aggregate(self.metrics, qrels_df, run_df)
        formatted_results = {}
        for metric, value in results.items():
            formatted_results[str(metric)] = value
            
        return formatted_results

