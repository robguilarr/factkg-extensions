# FactKG Extensions: A Research Proof-of-Concept

## Overview

This repository serves as a research Proof-of-Concept (PoC) accompanying an internship proposal. It is designed to extend and address specific limitations identified in the original research paper, "FactKG: Fact Verification via Reasoning on Knowledge Graphs" [1]. The original paper introduced a novel dataset of 108k claims and a baseline methodology for fact verification requiring reasoning over Knowledge Graphs (KGs). While groundbreaking, the original work left several low-hanging-fruit research gaps, which this repository aims to explore and implement.

**Disclaimer:** The implementations provided in this repository are currently at the Proof-of-Concept stage. The results and methodologies have not yet been validated at scale across the entire FactKG dataset.

## Repository Structure

The repository is structured to cleanly separate the new extension modules from the original codebase adapters, ensuring modularity and ease of testing.

```text
factkg-extensions/
├── conf/                       # YAML configuration files
│   ├── default.yaml
│   ├── gap1.yaml
│   ├── gap2.yaml
│   └── gap3.yaml
├── data/                       # Directory for datasets and pickles
├── docs/                       # Detailed documentation
│   ├── 00_original_paper.md
│   ├── 01_gap_one.md
│   ├── 02_gap_two.md
│   └── 03_gap_three.md
├── src/
│   └── factkg_ext/
│       ├── common/             # Shared utilities (e.g., LLM clients)
│       ├── constants/          # System-wide constants and prompts
│       ├── gap1_llm_rag/       # Implementation for Gap 1
│       ├── gap2_kg_update/     # Implementation for Gap 2
│       ├── gap3_dense_retrieval/ # Implementation for Gap 3
│       ├── original/           # Adapters for the original FactKG codebase
│       └── utilities/          # Configuration loaders
├── tests/                      # Pytest test suite
├── pyproject.toml              # Project metadata and dependencies
└── README.md                   # This file
```

## Quickstart

To get started with the `factkg-extensions` repository, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd factkg-extensions
    ```
2.  **Install the package and its dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -e ".[dev]"
    ```
3.  **Run the test suite:**
    Ensure everything is set up correctly by running the provided tests.
    ```bash
    pytest tests/
    ```

## Research Gaps Addressed

This PoC addresses three primary research gaps identified in the original FactKG paper.

### Gap 1: Evaluating Modern LLMs on FactKG

The original paper evaluated the dataset using text-only baselines like BERT and a zero-shot Flan-T5-XL model. This gap introduces a modern evaluation harness to benchmark state-of-the-art Large Language Models (LLMs) on FactKG. It supports both zero-shot and Retrieval-Augmented Generation (RAG) setups, allowing researchers to investigate whether massive LLMs have internalized the necessary knowledge or if they still struggle with complex multi-hop KG reasoning. For more details, see [docs/01_gap_one.md](docs/01_gap_one.md).

### Gap 2: Updating FactKG to Recent DBpedia/Wikidata Versions

A stated limitation of the original FactKG dataset is its reliance on the outdated 2015-10 version of DBpedia. This gap provides a pipeline to remap existing claims and entities to current DBpedia or Wikidata releases via SPARQL queries. By flagging stale facts, this extension introduces a temporal dimension to KG-based fact verification, making the benchmark more applicable to real-world, evolving knowledge bases. For more details, see [docs/02_gap_two.md](docs/02_gap_two.md).

### Gap 3: Improving Subgraph Retrieval with Dense Retrieval Methods

The original baseline utilizes a rigid Subgraph Retrieval Module consisting of a Relation Classifier and a Hop Classifier. This gap replaces that mechanism with a bi-encoder dense retriever (e.g., using SentenceTransformers). By scoring claim-vs-relation-path cosine similarity, this approach aims to significantly improve the recall of relevant evidence paths, particularly for claims written in a colloquial style that may not perfectly align with the KG's schema. For more details, see [docs/03_gap_three.md](docs/03_gap_three.md).

## Attribution and References

This work builds upon the foundational research presented in the FactKG paper.

*   **Original Paper:** Kim, J., et al. (2023). "FactKG: Fact Verification via Reasoning on Knowledge Graphs." *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023)*. [arXiv:2305.06590](https://arxiv.org/abs/2305.06590)
*   **Original Repository:** [https://github.com/jiho283/FactKG](https://github.com/jiho283/FactKG) (MIT License)
*   **Datasets:** The FactKG dataset is derived from the WebNLG dataset [2] and DBpedia [3].

**References:**
[1] Kim, J., et al. (2023). FactKG: Fact Verification via Reasoning on Knowledge Graphs. arXiv preprint arXiv:2305.06590.
[2] Gardent, C., et al. (2017). Creating Training Corpora for NLG Micro-Planners.
[3] Lehmann, J., et al. (2015). DBpedia - A large-scale, multilingual knowledge base extracted from Wikipedia.
