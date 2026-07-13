# FactKG Extensions

**Proof-of-concept research extensions for FactKG: fact verification via reasoning on knowledge graphs.**

This repository implements four novel research proposals that extend the FactKG framework (Kim et al., SNU). Each proposal addresses a specific limitation identified through deep analysis of the original paper and is grounded in recent literature (2024–2026).

> **Original Paper:** FactKG: Fact Verification via Reasoning on Knowledge Graphs  
> **Authors:** Jiho Kim, Sungjin Park, Yeonsu Kwon, Yohan Jo et al. (SNU)  
> **Link:** [arXiv:2305.06590](https://arxiv.org/abs/2305.06590)  
> **Original Repository:** [github.com/jiho283/FactKG](https://github.com/jiho283/FactKG)

---

## Overview

FactKG introduces a large-scale fact verification dataset (108K claims, 5 reasoning types) grounded in DBpedia, with baselines including BERT-based classifiers and graph attention networks (GEAR). This repository extends the framework with four research gaps:

| Gap | Module | Innovation | Key Reference |
|:---:|:-------|:-----------|:--------------|
| 1 | `gap1_llm_rag` | LLM-based verification with retrieval-augmented generation | — |
| 2 | `gap2_kg_update` | Temporal KG update pipeline (DBpedia 2015 → 2022+) | — |
| 3 | `gap3_dense_retrieval` | Bi-encoder dense retrieval for evidence subgraphs | — |
| 4 | `gap4_qgr_fingerprint` | Quantized evidence fingerprints for temporal-robust verification | QGR Survey (Chen et al., 2025) |

---

## Repository Structure

```text
.
├── README.md
├── pyproject.toml
├── conf/
│   ├── default.yaml
│   ├── gap1.yaml
│   ├── gap2.yaml
│   ├── gap3.yaml
│   └── gap4.yaml
├── docs/
│   ├── 00_original_paper.md
│   ├── 01_gap_one.md
│   ├── 02_gap_two.md
│   ├── 03_gap_three.md
│   ├── 04_gap_four.md
│   └── theory/
│       ├── vector_quantization.md
│       ├── fact_verification.md
│       └── knowledge_graph_evolution.md
├── src/
│   └── factkg_ext/
│       ├── __init__.py
│       ├── common/
│       ├── constants/
│       ├── utilities/
│       ├── original/
│       ├── gap1_llm_rag/
│       ├── gap2_kg_update/
│       ├── gap3_dense_retrieval/
│       └── gap4_qgr_fingerprint/
└── tests/
    ├── test_gap1.py
    ├── test_gap2.py
    ├── test_gap3.py
    └── test_gap4.py
```

---

## Implemented Extensions

### Gap 1: LLM-Based Verification with RAG

**Problem:** The original baselines (BERT, GEAR) require task-specific fine-tuning and cannot generalize to unseen reasoning patterns.

**Solution:** A `RAGVerifier` retrieves relevant KG subgraphs and passes them to an LLM with structured prompts for zero-shot fact verification, enabling generalization without retraining.

**Key formula:**

$$P(\text{label} | \text{claim}, \text{evidence}) = \text{LLM}(\text{prompt}(\text{claim}, \text{top-K triples}))$$

**Documentation:** [docs/01_gap_one.md](docs/01_gap_one.md)

---

### Gap 2: Temporal KG Update Pipeline

**Problem:** FactKG is grounded in a 2015 DBpedia snapshot. Many claims are now verifiable against updated knowledge that didn't exist in 2015.

**Solution:** A `KGUpdatePipeline` aligns the 2015 DBpedia entities with newer versions, identifies changed/added/removed triples, and re-evaluates claims against the updated graph.

**Key formula:**

$$\Delta_{\text{KG}} = \text{KG}_{2022} \setminus \text{KG}_{2015} \cup \text{KG}_{2015} \setminus \text{KG}_{2022}$$

**Documentation:** [docs/02_gap_two.md](docs/02_gap_two.md)

---

### Gap 3: Bi-Encoder Dense Retrieval

**Problem:** The original pipeline uses sparse keyword matching for evidence retrieval, missing semantically relevant triples.

**Solution:** A `DenseRetriever` uses a bi-encoder architecture (claim encoder + triple encoder) trained contrastively to retrieve evidence subgraphs based on semantic similarity.

**Key formula:**

$$\text{score}(c, t) = \frac{\text{enc}_c(c) \cdot \text{enc}_t(t)}{\|\text{enc}_c(c)\| \cdot \|\text{enc}_t(t)\|}$$

**Documentation:** [docs/03_gap_three.md](docs/03_gap_three.md)

---

### Gap 4: Quantized Evidence Fingerprints (QGR)

**Problem:** When the KG updates, continuous embeddings of all connected entities must be recomputed. This makes the system brittle to temporal changes.

**Solution:** Both claims and evidence subgraphs are encoded as discrete code fingerprints. Verification becomes code alignment (Jaccard similarity). Incremental updates affect only directly changed entities — $O(|\text{changed}|)$ instead of $O(|\text{all}|)$.

**Key formulas:**

$$A(\text{claim}, \text{evidence}) = \frac{\sum_{i=1}^{L} w_i \cdot \mathbb{1}[c_i = c_i']}{\ \sum_{i=1}^{L} w_i}$$

$$\mathbf{h}_{G_e}^{\text{new}} = \frac{m \cdot \mathbf{h}_{G_e}^{\text{old}} - \sum_{\text{removed}} \mathbf{h}_i + \sum_{\text{added}} \mathbf{h}_j}{m - |\text{removed}| + |\text{added}|}$$

**Documentation:** [docs/04_gap_four.md](docs/04_gap_four.md)

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/robguilarr/factkg-extensions.git
cd factkg-extensions
pip install -e ".[dev]"

# Run all tests (offline, no API key needed)
pytest tests/ -v

# Run a specific gap's tests
pytest tests/test_gap4.py -v
```

---

## Configuration

Each gap has its own YAML configuration in `conf/`. Load configurations with:

```python
from factkg_ext.utilities.config import load_config
config = load_config("conf/gap4.yaml")
```

---

## Dependencies

| Group | Packages | Purpose |
|:------|:---------|:--------|
| Core | `numpy`, `pyyaml`, `pydantic`, `networkx`, `requests` | Base functionality |
| Dev | `pytest`, `ruff` | Testing and linting |
| ML | `torch`, `torch-geometric`, `sentence-transformers`, `openai` | Full pipeline (optional) |

---

## Theory

Foundational concepts and source papers are documented in `docs/theory/`:

- [Vector Quantization](docs/theory/vector_quantization.md) — Multi-head VQ, code fingerprints, incremental updates
- [Fact Verification](docs/theory/fact_verification.md) — FEVER task, claim decomposition, evidence retrieval, label prediction
- [Knowledge Graph Evolution](docs/theory/knowledge_graph_evolution.md) — Temporal KGs, DBpedia versioning, entity alignment

---

## Disclaimer

The implementations provided herein are proof-of-concept prototypes intended to demonstrate the feasibility of the proposed extensions. Results discussed in the documentation are theoretical or based on limited preliminary testing and have not yet been validated at scale across all benchmark datasets.

---

## References

1. Kim, J. et al. (2023). FactKG: Fact Verification via Reasoning on Knowledge Graphs. ACL 2023.
2. Chen et al. (2025). A Survey on Quantized Graph Representation. arXiv:2502.00681.
3. van den Oord et al. (2017). Neural Discrete Representation Learning (VQ-VAE). NeurIPS 2017.
4. Zhou et al. (2019). GEAR: Graph-based Evidence Aggregating and Reasoning. ACL 2019.
5. Thorne et al. (2018). FEVER: A Large-scale Dataset for Fact Extraction and VERification. NAACL 2018.
