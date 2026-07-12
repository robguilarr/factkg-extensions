# Gap 4: Quantized Evidence Fingerprints for Temporal-Robust Fact Verification

## Motivation

FactKG verifies claims against a static snapshot of DBpedia (2015 dump). When the knowledge graph updates — entities change, relations are added or removed — the entire evidence retrieval and verification pipeline must be re-executed. This is because continuous embeddings are holistic: changing one triple affects the embedding of all connected entities through message passing.

**Quantized evidence fingerprints** solve this by representing both claims and evidence subgraphs as discrete code sequences. The key advantages:

1. **Temporal robustness**: When the KG updates, only the codes of directly affected entities need re-quantization. Unaffected regions retain their codes exactly.
2. **Incremental updates**: Adding/removing a triple requires $O(1)$ code updates instead of $O(n)$ re-embedding.
3. **Interpretable verification**: Code alignment between claim and evidence provides a transparent similarity score, unlike black-box neural classifiers.
4. **Efficient storage**: Code fingerprints require 16 bytes per claim/evidence vs. 1024 bytes for continuous embeddings.

This directly addresses FactKG's limitation of being tied to a 2015 DBpedia snapshot — with code fingerprints, the system can track KG evolution incrementally.

## Mathematical Formulation

### Claim Fingerprinting

A claim text $t$ is encoded to a dense embedding then quantized:

$$\mathbf{h}_t = \text{Encoder}(t) \in \mathbb{R}^d$$

$$\text{FP}(t) = \text{VQ}(\mathbf{h}_t) = [c_1, c_2, \ldots, c_L]$$

where $\text{VQ}$ is multi-head vector quantization with $L$ heads.

### Evidence Fingerprinting

An evidence subgraph $G_e = \{(s_i, p_i, o_i)\}_{i=1}^{m}$ is encoded via triple aggregation:

$$\mathbf{h}_{G_e} = \frac{1}{m} \sum_{i=1}^{m} \frac{\mathbf{e}(s_i) + \mathbf{e}(p_i) + \mathbf{e}(o_i)}{\|\mathbf{e}(s_i) + \mathbf{e}(p_i) + \mathbf{e}(o_i)\|}$$

$$\text{FP}(G_e) = \text{VQ}(\mathbf{h}_{G_e}) = [c_1', c_2', \ldots, c_L']$$

### Code Alignment Score

Verification is reduced to code comparison:

$$A(\text{claim}, \text{evidence}) = \frac{\sum_{i=1}^{L} w_i \cdot \mathbb{1}[c_i = c_i']}{\ \sum_{i=1}^{L} w_i}$$

where $w_i$ are learned or uniform position weights.

### Verification Decision

$$\text{Label} = \begin{cases} \text{SUPPORTS} & \text{if } A \geq \theta_s \\ \text{REFUTES} & \text{if } A \leq \theta_r \\ \text{NOT\_ENOUGH\_INFO} & \text{otherwise} \end{cases}$$

with default thresholds $\theta_s = 0.5$, $\theta_r = 0.2$.

### Incremental Update

When triples are added/removed from the evidence:

$$\mathbf{h}_{G_e}^{\text{new}} = \frac{m \cdot \mathbf{h}_{G_e}^{\text{old}} - \sum_{\text{removed}} \mathbf{h}_i + \sum_{\text{added}} \mathbf{h}_j}{m - |\text{removed}| + |\text{added}|}$$

Then re-quantize: $\text{FP}(G_e^{\text{new}}) = \text{VQ}(\mathbf{h}_{G_e}^{\text{new}})$

Cost: $O(|\text{changed triples}|)$ instead of $O(|\text{all triples}|)$.

### Temporal Stability Metric

$$\Delta_{\text{temporal}} = |A(\text{claim}, G_e^{t_1}) - A(\text{claim}, G_e^{t_2})|$$

Lower $\Delta$ = more stable verification across KG versions.

## Implementation Walkthrough

### `fingerprinter.py` — ClaimFingerprinter

Encodes claim text into code fingerprints:

- `_text_to_embedding(text)`: Converts text to dense vector (hash-based for offline; sentence-transformers in production).
- `fingerprint(claim_text)`: Full pipeline: embed → quantize → return `Fingerprint` with codes and metadata.
- `batch_fingerprint(claims)`: Processes multiple claims.

### `fingerprinter.py` — EvidenceFingerprinter

Encodes evidence subgraphs:

- `_encode_triple(triple)`: TransE-style additive encoding with normalization.
- `fingerprint(evidence_triples)`: Mean-pool triple embeddings, normalize, quantize.
- `incremental_update(old_fp, removed, added, total)`: Efficient update by subtracting removed and adding new triple contributions without full recomputation.

### `verifier.py` — CodeAlignmentVerifier

The verification decision layer:

- `compute_alignment(claim_codes, evidence_codes)`: Weighted or uniform Jaccard similarity.
- `verify(claim_text, evidence_triples)`: End-to-end verification returning label + confidence.
- `compute_temporal_stability(claim, evidence_before, evidence_after)`: Measures how much the verification score changes across KG versions.

## Configuration

See `conf/gap4.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `codebook_size` | 512 | Entries per head codebook |
| `code_length` | 16 | Number of code positions |
| `support_threshold` | 0.5 | Score above → SUPPORTS |
| `refute_threshold` | 0.2 | Score below → REFUTES |
| `incremental_update` | true | Enable incremental re-quantization |
| `use_weighted_alignment` | true | Use position-weighted alignment |

## Evaluation Plan

### Tier 1: Codebook Health
- **Perplexity** > 256 (50% of codebook size)
- **Reconstruction MSE** < 0.04 on DBpedia entity embeddings
- **Incremental update fidelity**: MSE between incremental and full recompute < 0.01

### Tier 2: Core Evaluation
- **Dataset**: FactKG (108K claims, 5 reasoning types)
- **Baseline**: GEAR (78.2% accuracy), FactKG-BERT (72.4%)
- **Success**: Per-claim-type accuracy maintained within 2%, incremental update time < 60s per 1K new facts
- **Key metric**: Temporal stability $\Delta < 0.05$ across DBpedia 2015 → 2022

### Tier 3: Ablation
- Code length: 8, 16, 32 (compression vs. discrimination)
- Threshold tuning: grid search over $(\theta_s, \theta_r)$
- Weighted vs. uniform alignment
- Negation handling: complement codes for negated claims
- Per-reasoning-type analysis (one-hop, multi-hop, conjunction, existence, negation)

### Known Failure Mode: Negation Blindness
Negated claims ("X did NOT do Y") may produce similar fingerprints to affirmed claims ("X did Y") because the structural evidence is identical — only the polarity differs. If negation accuracy drops below 60%, implement **complement codes**: flip all code indices for negated claims before alignment.
