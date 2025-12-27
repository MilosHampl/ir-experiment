# IR Experiment

Comparative experiment of retrieval methods based on the Cranfield paradigm using the `beir/trec-covid` dataset.

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
poetry install
```

## Usage

Run the full experiment pipeline:

```bash
poetry run python run_experiment.py
```

This will:
1. Load the dataset.
2. Generate/load embeddings (cached in `cache/`).
3. Run five retrieval methods:
   - Sparse (BM25 with preprocessing)
   - Sparse (BM25 without preprocessing)
   - Dense (HNSW approximate nearest neighbor)
   - Hybrid (RRF fusion of sparse and dense)
   - Hybrid + Reranking (cross-encoder reranking)
4. Evaluate results and save them to `experiment_results.csv`.
