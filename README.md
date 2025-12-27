# IR Experiment

Comparative experiment of retrieval methods based on the Cranfield paradigm using the `beir/trec-covid` dataset.

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
# Install dependencies
poetry install
```

## Usage

Run the full experiment pipeline:

```bash
poetry run python run_experiment.py
```

This will:
1. Load the dataset.
2. Generate/Load embeddings (cached in `cache/`).
3. Run Sparse (BM25), Dense (Exact), Hybrid (RRF), and Reranking retrieval.
4. Evaluate results and save them to `experiment_results.csv`.
