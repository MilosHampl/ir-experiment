import logging
import pandas as pd
from tqdm import tqdm
from src.data_loader import DataLoader
from src.retrievers.sparse import SparseRetriever
from src.retrievers.dense import DenseRetriever
from src.retrievers.hybrid import perform_rrf
from src.retrievers.reranker import Reranker
from src.evaluation import Evaluator

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting RAG Experiment...")

    # 1. Load Data
    loader = DataLoader()
    corpus = loader.load_corpus()
    queries = loader.load_queries()
    qrels = loader.load_qrels()

    # 2. Initialize Retrievers
    logger.info("Initializing Retrievers...")
    
    # Sparse (Standard Preprocessing)
    sparse_retriever = SparseRetriever(corpus, preprocess=True)
    
    # Sparse (No Preprocessing)
    sparse_retriever_no_preproc = SparseRetriever(corpus, preprocess=False)
    
    # Dense (Handles Embeddings & Indexing)
    dense_retriever = DenseRetriever(corpus)
    
    # Reranker
    reranker = Reranker(corpus)
    
    # Evaluator
    evaluator = Evaluator(qrels)

    # Storage for run results
    results_sparse = {}
    results_sparse_no_preproc = {}
    results_dense = {}
    results_hybrid = {}
    results_rerank = {}

    # 3. Execute Runs
    logger.info("Executing retrieval runs...")
    
    # Iterate over queries
    # For testing purposes, you might want to limit queries, but for thesis, run all.
    # trec-covid has 50 queries.
    
    for _, row in tqdm(queries.iterrows(), total=len(queries), desc="Processing Queries"):
        query_id = str(row['query_id'])
        query_text = row['text']
        
        # A. Sparse Run
        sparse_res = sparse_retriever.search(query_text, top_k=1000)
        results_sparse[query_id] = sparse_res
        
        # A2. Sparse Run (No Preproc)
        sparse_res_no_preproc = sparse_retriever_no_preproc.search(query_text, top_k=1000)
        results_sparse_no_preproc[query_id] = sparse_res_no_preproc
        
        # B. Dense Run (Exact)
        dense_res = dense_retriever.search(query_text, top_k=1000)
        results_dense[query_id] = dict(dense_res) # Convert list of tuples back to dict for evaluation/RRF
        
        # C. Hybrid Run (RRF)
        # RRF expects dicts
        hybrid_res = perform_rrf(sparse_res, dict(dense_res))
        results_hybrid[query_id] = hybrid_res
        
        # D. Rerank Run (Top 50 from Hybrid)
        rerank_res = reranker.rerank(query_text, hybrid_res)
        results_rerank[query_id] = rerank_res

    # 4. Evaluate
    logger.info("Evaluating runs...")
    
    metrics_sparse = evaluator.evaluate(results_sparse)
    metrics_sparse_no_preproc = evaluator.evaluate(results_sparse_no_preproc)
    metrics_dense = evaluator.evaluate(results_dense)
    metrics_hybrid = evaluator.evaluate(results_hybrid)
    metrics_rerank = evaluator.evaluate(results_rerank)

    # 5. Compare Results
    comparison_df = pd.DataFrame([
        {"Method": "Sparse (BM25)", **metrics_sparse},
        {"Method": "Sparse (BM25) No Preproc", **metrics_sparse_no_preproc},
        {"Method": "Dense (Exact)", **metrics_dense},
        {"Method": "Hybrid (RRF)", **metrics_hybrid},
        {"Method": "Hybrid + Rerank", **metrics_rerank}
    ])
    
    # Rename metrics to match requirements
    comparison_df.rename(columns={
        "AP": "MAP",
        "RR": "MRR"
    }, inplace=True)
    
    # Reorder columns for better readability
    cols = ["Method", "nDCG@10", "P@10", "MAP", "MRR", "R@100"]
    # Filter cols that exist
    cols = [c for c in cols if c in comparison_df.columns]
    comparison_df = comparison_df[cols]

    print("\n=== Experiment Results ===")
    print(comparison_df.to_string(index=False))
    
    # Save results
    comparison_df.to_csv("experiment_results.csv", index=False)
    logger.info("Experiment completed. Results saved to experiment_results.csv")

if __name__ == "__main__":
    main()

