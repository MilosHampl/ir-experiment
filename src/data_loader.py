import ir_datasets
import pandas as pd
from typing import Dict, Tuple, List
import logging
from config import DATASET_NAME

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, dataset_name: str = DATASET_NAME):
        self.dataset_name = dataset_name
        self.dataset = ir_datasets.load(dataset_name)
        logger.info(f"Initialized DataLoader for {dataset_name}")

    def load_corpus(self) -> pd.DataFrame:
        """
        Loads the corpus (documents).
        Returns a DataFrame with columns ['doc_id', 'text', 'title'].
        """
        logger.info("Loading corpus...")
        docs_iter = self.dataset.docs_iter()
        
        data = []
        for doc in docs_iter:
            data.append({
                "doc_id": doc.doc_id,
                "text": doc.text,
                "title": doc.title
            })
            
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} documents.")
        return df

    def load_queries(self) -> pd.DataFrame:
        """
        Loads the queries (topics).
        Returns a DataFrame with columns ['query_id', 'text'].
        """
        logger.info("Loading queries...")
        queries_iter = self.dataset.queries_iter()
        
        data = []
        for query in queries_iter:
            data.append({
                "query_id": query.query_id,
                "text": query.text
            })
            
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} queries.")
        return df

    def load_qrels(self) -> pd.DataFrame:
        """
        Loads the qrels (relevance judgments).
        Returns a DataFrame with columns ['query_id', 'doc_id', 'relevance', 'iteration'].
        """
        logger.info("Loading qrels...")
        qrels_iter = self.dataset.qrels_iter()
        
        data = []
        for qrel in qrels_iter:
            data.append({
                "query_id": qrel.query_id,
                "doc_id": qrel.doc_id,
                "relevance": qrel.relevance,
                "iteration": qrel.iteration
            })
            
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} qrels.")
        return df

