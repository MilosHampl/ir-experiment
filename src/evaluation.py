import logging
import pandas as pd
import ir_measures
from ir_measures import nDCG, R, AP, P, RR
from typing import Dict

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, qrels: pd.DataFrame):
        self.qrels = qrels
        self.metrics = [nDCG@10, R@100, R@500, AP, P@10, RR]

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

    def get_per_query_metrics(self, run_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        run_data = []
        for qid, docs in run_results.items():
            for doc_id, score in docs.items():
                run_data.append({
                    "query_id": str(qid),
                    "doc_id": str(doc_id),
                    "score": score
                })
        
        if not run_data:
            return pd.DataFrame()
            
        run_df = pd.DataFrame(run_data)
        
        qrels_df = self.qrels.copy()
        qrels_df['query_id'] = qrels_df['query_id'].astype(str)
        qrels_df['doc_id'] = qrels_df['doc_id'].astype(str)
        
        # iter_calc returns an iterator of (metric, query_id, val)
        per_query_results = []
        for metric in self.metrics:
            for result in ir_measures.iter_calc([metric], qrels_df, run_df):
                per_query_results.append({
                    "query_id": result.query_id,
                    "metric": str(result.measure),
                    "value": result.value
                })
                
        df = pd.DataFrame(per_query_results)
        # Pivot so we have query_id as index and metrics as columns
        pivot_df = df.pivot(index='query_id', columns='metric', values='value')
        return pivot_df
