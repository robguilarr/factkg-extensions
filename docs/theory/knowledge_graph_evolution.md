# Knowledge Graph Evolution: Theory and Foundations

## Definition

**Knowledge Graph Evolution** refers to the process by which knowledge graphs change over time — entities are added/removed, relations are updated, and facts become outdated. Systems that depend on KGs must handle these changes gracefully without full retraining.

---

## Core Concepts

### 1. Types of KG Changes

| Change Type | Example | Frequency |
|:------------|:--------|:----------|
| Entity addition | New person, organization, or event | High |
| Entity removal | Merged/deprecated entities | Low |
| Relation addition | New fact discovered | High |
| Relation removal | Fact becomes false | Medium |
| Relation update | Value changes (e.g., population) | Medium |

### 2. DBpedia Versioning

DBpedia releases periodic snapshots extracted from Wikipedia. Key versions:

| Version | Year | Entities | Triples |
|:--------|:----:|:--------:|:-------:|
| DBpedia 2015-10 | 2015 | ~4.6M | ~580M |
| DBpedia 2016-10 | 2016 | ~5.0M | ~620M |
| DBpedia 2022-03 | 2022 | ~6.2M | ~800M |

FactKG uses the 2015 snapshot. Between 2015 and 2022, approximately 30% of triples changed.

> **Source:** DBpedia Association. [https://www.dbpedia.org/](https://www.dbpedia.org/)

### 3. Entity Alignment Across Versions

When the KG updates, entity URIs may change. Entity alignment maps old URIs to new ones:

$$\text{align}: \text{URI}_{t_1} \rightarrow \text{URI}_{t_2}$$

Methods:
- **URI matching:** Same URI = same entity (works for ~90% of cases)
- **Label matching:** Same rdfs:label = likely same entity
- **Embedding alignment:** Nearest neighbor in embedding space

### 4. Incremental Embedding Updates

Traditional approach: retrain all embeddings when KG changes. Incremental approach:

$$\mathbf{h}_e^{\text{new}} = \mathbf{h}_e^{\text{old}} + \Delta \mathbf{h}_e$$

where $\Delta \mathbf{h}_e$ is computed only from the changed neighborhood.

**Continual KG Embedding (CKGE):**

$$\mathcal{L}_{\text{CKGE}} = \mathcal{L}_{\text{new triples}} + \lambda \cdot \mathcal{L}_{\text{distillation}}(\theta_{\text{new}}, \theta_{\text{old}})$$

> **Source:** Cui, J. et al. (2025). PS-CKGE: Parameter-Efficient Continual Knowledge Graph Embedding. *KDD 2025*.

### 5. Quantized Incremental Updates

With code-based representations, updates are even cheaper:

$$\text{FP}(G_e^{\text{new}}) = \text{VQ}\left(\frac{m \cdot \mathbf{h}^{\text{old}} - \sum_{\text{removed}} + \sum_{\text{added}}}{m + |\text{added}| - |\text{removed}|}\right)$$

Only the codes of directly affected entities change. Unaffected regions retain their codes exactly — a key advantage over continuous embeddings where changes propagate through message passing.

---

## Relevance to This Repository

In FactKG extensions:

1. **Gap 2 (KG Update):** Implements the full DBpedia 2015 → 2022 alignment pipeline.
2. **Gap 4 (QGR Fingerprint):** Incremental code updates enable temporal-robust verification without full recomputation.

---

## Key Source Papers

| Paper | Year | Venue | Contribution |
|:------|:----:|:-----:|:-------------|
| DBpedia | 2007 | ISWC | Structured knowledge from Wikipedia |
| PS-CKGE | 2025 | KDD | Continual KG embedding without forgetting |
| RAKEL | 2026 | arXiv | Graph transformer knowledge editing |
| GraIL | 2020 | ICML | Inductive relation prediction on new entities |
| NodePiece | 2022 | ICLR | Compositional entity representations |
