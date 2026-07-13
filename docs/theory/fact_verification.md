# Fact Verification: Theory and Foundations

## Definition

**Fact Verification** is the task of determining whether a natural language claim is supported, refuted, or unverifiable given a knowledge source. In the FactKG context, the knowledge source is a structured knowledge graph (DBpedia), and verification requires multi-hop reasoning over graph paths.

---

## Core Concepts

### 1. FEVER Task Formulation

The standard fact verification task (Thorne et al., 2018) classifies claims into three labels:

- **SUPPORTS:** Evidence confirms the claim
- **REFUTES:** Evidence contradicts the claim
- **NOT ENOUGH INFO:** Insufficient evidence to decide

> **Source:** Thorne, J. et al. (2018). FEVER: a Large-scale Dataset for Fact Extraction and VERification. *NAACL 2018*. [arXiv:1803.05355](https://arxiv.org/abs/1803.05355)

### 2. FactKG Reasoning Types

FactKG extends standard fact verification with five reasoning types grounded in KG structure:

| Type | Description | Example |
|:-----|:------------|:--------|
| One-hop | Single triple verification | "Paris is in France" |
| Multi-hop | Path traversal (2+ hops) | "The director of Inception studied at UCL" |
| Conjunction | Multiple independent facts | "X is a Y and Z" |
| Existence | Entity/relation existence | "There exists a movie directed by X" |
| Negation | Negated claims | "X did NOT win the award" |

> **Source:** Kim, J. et al. (2023). FactKG: Fact Verification via Reasoning on Knowledge Graphs. *ACL 2023*. [arXiv:2305.06590](https://arxiv.org/abs/2305.06590)

### 3. GEAR (Graph-based Evidence Aggregating and Reasoning)

GEAR uses a graph attention network to aggregate evidence from multiple retrieved sentences/triples:

$$\mathbf{h}_i^{(l+1)} = \text{ReLU}\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W} \mathbf{h}_j^{(l)}\right)$$

where $\alpha_{ij}$ are attention weights between evidence nodes.

> **Source:** Zhou, J. et al. (2019). GEAR: Graph-based Evidence Aggregating and Reasoning for Fact Verification. *ACL 2019*. [arXiv:1908.01843](https://arxiv.org/abs/1908.01843)

### 4. Evidence Retrieval for Verification

The retrieval stage finds relevant KG subgraphs for a given claim:

1. **Entity linking:** Map claim entities to KG nodes
2. **Subgraph extraction:** Retrieve k-hop neighborhood around linked entities
3. **Path ranking:** Score paths by relevance to the claim
4. **Evidence selection:** Select top-N paths as evidence

### 5. Temporal Robustness

A key challenge: KGs evolve over time. A claim verified as TRUE against DBpedia 2015 might be FALSE against DBpedia 2022 (e.g., "X is the president of Y" changes with elections).

**Temporal stability metric:**

$$\Delta_{\text{temporal}} = |P(\text{label} | \text{claim}, \text{KG}_{t_1}) - P(\text{label} | \text{claim}, \text{KG}_{t_2})|$$

Low $\Delta$ indicates the verification system is robust to benign KG updates.

---

## Evaluation Metrics

| Metric | Formula | Use Case |
|:-------|:--------|:---------|
| Accuracy | $\frac{\text{correct}}{\text{total}}$ | Overall performance |
| Per-type accuracy | Accuracy within each reasoning type | Fine-grained analysis |
| Label-weighted F1 | Macro F1 across 3 labels | Class-imbalanced datasets |
| FEVER Score | Accuracy with correct evidence | Joint retrieval + verification |

---

## Relevance to This Repository

In FactKG extensions:

1. **Gap 1 (LLM RAG):** Zero-shot verification using retrieved evidence.
2. **Gap 2 (KG Update):** Temporal alignment across DBpedia versions.
3. **Gap 3 (Dense Retrieval):** Bi-encoder evidence retrieval.
4. **Gap 4 (QGR Fingerprint):** Code-based verification robust to KG evolution.

---

## Key Source Papers

| Paper | Year | Venue | Contribution |
|:------|:----:|:-----:|:-------------|
| FEVER | 2018 | NAACL | Large-scale fact verification dataset |
| FactKG | 2023 | ACL | KG-grounded fact verification with reasoning types |
| GEAR | 2019 | ACL | Graph attention for evidence aggregation |
| KGAT | 2020 | ACL | Knowledge graph attention for verification |
| ProgramFC | 2023 | ACL | Program-guided fact checking |
| ProoFVer | 2022 | NAACL | Natural logic proof for verification |
