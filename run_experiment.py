import logging
import pandas as pd
from tqdm import tqdm
from scipy.stats import wilcoxon
from src.data_loader import DataLoader
from src.retrievers.sparse import SparseRetriever
from src.retrievers.dense import DenseRetriever
from src.retrievers.hybrid import perform_rrf
from src.retrievers.reranker import Reranker
from src.evaluation import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting RAG Experiment...")

    loader = DataLoader()
    corpus = loader.load_corpus()
    queries = loader.load_queries()
    qrels = loader.load_qrels()

    logger.info("Initializing Retrievers...")
    sparse_retriever = SparseRetriever(corpus, preprocess=True)
    sparse_retriever_no_preproc = SparseRetriever(corpus, preprocess=False)
    dense_retriever = DenseRetriever(corpus)
    reranker = Reranker(corpus)
    evaluator = Evaluator(qrels)
    results_sparse = {}
    results_sparse_no_preproc = {}
    results_dense = {}
    results_hybrid = {}
    results_hybrid_no_preproc = {}
    results_rerank = {}

    logger.info("Executing retrieval runs...")
    for _, row in tqdm(queries.iterrows(), total=len(queries), desc="Processing Queries"):
        query_id = str(row['query_id'])
        query_text = row['text']

        sparse_res = sparse_retriever.search(query_text, top_k=1000)
        results_sparse[query_id] = sparse_res
        
        sparse_res_no_preproc = sparse_retriever_no_preproc.search(query_text, top_k=1000)
        results_sparse_no_preproc[query_id] = sparse_res_no_preproc
        
        dense_res = dense_retriever.search(query_text, top_k=1000)
        results_dense[query_id] = dense_res
        
        hybrid_res = perform_rrf(sparse_res, dense_res)
        results_hybrid[query_id] = hybrid_res
        
        rerank_res = reranker.rerank(query_text, hybrid_res)
        results_rerank[query_id] = rerank_res

    logger.info("Evaluating runs...")
    metrics_sparse = evaluator.evaluate(results_sparse)
    metrics_sparse_no_preproc = evaluator.evaluate(results_sparse_no_preproc)
    metrics_dense = evaluator.evaluate(results_dense)
    metrics_hybrid = evaluator.evaluate(results_hybrid)
    metrics_rerank = evaluator.evaluate(results_rerank)
    comparison_df = pd.DataFrame([
        {"Method": "Sparse (BM25)", **metrics_sparse},
        {"Method": "Sparse (BM25) No Preproc", **metrics_sparse_no_preproc},
        {"Method": "Dense (HNSW)", **metrics_dense},
        {"Method": "Hybrid (RRF)", **metrics_hybrid},
        {"Method": "Hybrid + Rerank", **metrics_rerank}
    ])
    
    comparison_df.rename(columns={
        "AP": "MAP",
        "RR": "MRR"
    }, inplace=True)
    
    cols = ["Method", "nDCG@10", "P@10", "MAP", "MRR", "R@100", "R@500"]
    cols = [c for c in cols if c in comparison_df.columns]
    comparison_df = comparison_df[cols]

    print("\n=== Experiment Results ===")
    print(comparison_df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
    
    comparison_df.to_csv("experiment_results.csv", index=False, float_format="%.4f")
    logger.info("Experiment completed. Results saved to experiment_results.csv")

    logger.info("Calculating per-query metrics for significance testing...")
    pq_sparse = evaluator.get_per_query_metrics(results_sparse)
    pq_sparse_no_preproc = evaluator.get_per_query_metrics(results_sparse_no_preproc)
    pq_dense = evaluator.get_per_query_metrics(results_dense)
    pq_hybrid = evaluator.get_per_query_metrics(results_hybrid)
    pq_rerank = evaluator.get_per_query_metrics(results_rerank)

    comparisons = [
        ("BM25", "BM25 Preproc", pq_sparse_no_preproc, pq_sparse, ["P@10", "nDCG@10", "AP"]),
        ("Hybrid", "BM25", pq_hybrid, pq_sparse, ["R@100", "RR"]),
        ("Hybrid", "HNSW", pq_hybrid, pq_dense, ["R@100", "RR"]),
        ("Rerank", "Hybrid", pq_rerank, pq_hybrid, ["nDCG@10", "P@10"]),
        ("Rerank", "HNSW", pq_rerank, pq_dense, ["nDCG@10", "P@10"]),
    ]

    sig_results = []
    
    for sys1, sys2, df1, df2, metrics in comparisons:
        # Align dataframes on query_id index
        joined = df1.join(df2, lsuffix='_1', rsuffix='_2', how='inner')
        
        for metric in metrics:
            col1 = f"{metric}_1"
            col2 = f"{metric}_2"
            
            if col1 not in joined.columns or col2 not in joined.columns:
                logger.warning(f"Metric {metric} missing for comparison {sys1} vs {sys2}")
                continue
                
            scores1 = joined[col1]
            scores2 = joined[col2]
            
            try:
                stat, p_value = wilcoxon(scores1, scores2, zero_method='pratt')
                sig_results.append({
                    "System A": sys1,
                    "System B": sys2,
                    "Metric": metric,
                    "p-value": p_value
                })
            except Exception as e:
                logger.warning(f"Error calculating Wilcoxon for {sys1} vs {sys2} ({metric}): {e}")
                sig_results.append({
                    "System A": sys1,
                    "System B": sys2,
                    "Metric": metric,
                    "p-value": None
                })
                
    if sig_results:
        sig_df = pd.DataFrame(sig_results)
        print("\n=== Significance Test Results (Wilcoxon) ===")
        print(sig_df.to_string(index=False, float_format=lambda x: "{:.15f}".format(x).rstrip('0').rstrip('.')))
        sig_df.to_csv("significance_results.csv", index=False, float_format="%.15f")
        logger.info("Significance results saved to significance_results.csv")

if __name__ == "__main__":
    main()

