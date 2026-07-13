# Vector Quantization: Theory and Foundations

## Definition

**Vector Quantization (VQ)** is a signal processing technique that maps continuous vectors to a finite set of discrete codebook entries. In the context of graph-LLM integration, VQ enables encoding graph-structural information as discrete tokens that LLMs can process natively.

---

## Core Concepts

### 1. Codebook

A codebook $\mathcal{C} = \{\mathbf{c}_1, \mathbf{c}_2, \ldots, \mathbf{c}_K\}$ is a set of $K$ learnable prototype vectors in $\mathbb{R}^d$. Each input vector is mapped to its nearest codebook entry:

$$q(\mathbf{x}) = \arg\min_{j \in \{1, \ldots, K\}} \|\mathbf{x} - \mathbf{c}_j\|^2$$

### 2. VQ-VAE (Vector Quantized Variational Autoencoder)

Introduced by van den Oord et al. (2017), VQ-VAE replaces the continuous latent space of a VAE with discrete codes. The encoder output is quantized to the nearest codebook entry, and the decoder reconstructs from the quantized representation.

**Training loss:**

$$\mathcal{L} = \|\mathbf{x} - \text{decode}(q(\text{encode}(\mathbf{x})))\|^2 + \|\text{sg}[\mathbf{z}_e] - \mathbf{c}\|^2 + \beta \|\mathbf{z}_e - \text{sg}[\mathbf{c}]\|^2$$

where $\text{sg}[\cdot]$ is the stop-gradient operator and $\beta$ is the commitment cost.

> **Source:** van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning. *NeurIPS 2017*. [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)

### 3. Residual Vector Quantization (RVQ)

RVQ applies multiple quantization stages sequentially, where each stage quantizes the residual error from the previous stage:

$$\mathbf{r}_0 = \mathbf{x}, \quad c_s = q_s(\mathbf{r}_{s-1}), \quad \mathbf{r}_s = \mathbf{r}_{s-1} - \mathbf{c}_{s, c_s}$$

The final reconstruction is: $\hat{\mathbf{x}} = \sum_{s=1}^{S} \mathbf{c}_{s, c_s}$

RVQ achieves higher fidelity than single-stage VQ with the same total codebook size because each stage captures progressively finer details.

> **Source:** Zeghidour, N. et al. (2021). SoundStream: An End-to-End Neural Audio Codec. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*. [arXiv:2107.03312](https://arxiv.org/abs/2107.03312)

### 4. Multi-Head Vector Quantization

Instead of quantizing the full vector at once, split it into $L$ chunks and quantize each independently:

$$\mathbf{x} = [\mathbf{x}_1, \ldots, \mathbf{x}_L], \quad c_i = q_i(\mathbf{x}_i)$$

This is equivalent to product quantization and allows each head to specialize in different aspects of the representation.

> **Source:** Jegou, H., Douze, M., & Schmid, C. (2011). Product Quantization for Nearest Neighbor Search. *IEEE TPAMI*.

### 5. EMA Codebook Updates

Exponential Moving Average (EMA) updates avoid the need for gradient-based codebook learning:

$$N_j \leftarrow \gamma N_j + (1 - \gamma) n_j$$

$$\mathbf{m}_j \leftarrow \gamma \mathbf{m}_j + (1 - \gamma) \sum_{i: q(i)=j} \mathbf{x}_i$$

$$\mathbf{c}_j \leftarrow \frac{\mathbf{m}_j}{N_j}$$

where $\gamma \in [0.99, 0.999]$ is the decay rate.

### 6. Straight-Through Estimator (STE)

Since $\arg\min$ is non-differentiable, gradients are passed through the quantization step unchanged:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_e} \approx \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{z}}}$$

This allows end-to-end training despite the discrete bottleneck.

> **Source:** Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or Propagating Gradients Through Stochastic Neurons. *arXiv:1308.3432*.

---

## Codebook Health Metrics

### Perplexity

Measures how uniformly the codebook is utilized:

$$\text{Perplexity} = \exp\left(-\sum_{j=1}^{K} p_j \log p_j\right), \quad p_j = \frac{n_j}{\sum_k n_k}$$

- **Healthy:** Perplexity > 50% of $K$ (diverse usage)
- **Collapsed:** Perplexity < 10% of $K$ (most codes unused)

### Dead Code Fraction

Percentage of codebook entries that are never or rarely assigned:

$$\text{DeadFraction} = \frac{|\{j : n_j < \epsilon\}|}{K}$$

Target: < 10%. Mitigation: codebook reset (re-initialize dead codes from encoder outputs).

### Reconstruction MSE

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2$$

Lower is better. Typical target: < 0.05 for normalized embeddings.

---

## Application to Graph-LLM Integration

In the QGR (Quantized Graph Representation) paradigm, VQ serves as the bridge between continuous graph embeddings and discrete LLM tokens:

1. **Graph Encoder** → continuous embedding $\mathbf{h} \in \mathbb{R}^d$
2. **VQ Layer** → discrete codes $[c_1, \ldots, c_L]$
3. **Dr.E Alignment** → vocabulary words $[w_1, \ldots, w_L]$
4. **LLM** → processes words using pre-trained language understanding

This eliminates the modality gap without adapter layers or projection heads.

> **Source:** Chen et al. (2025). A Survey on Quantized Graph Representation. *arXiv:2502.00681*. [Link](https://arxiv.org/abs/2502.00681)

---

## Key Source Papers

| Paper | Year | Venue | Contribution |
|:------|:----:|:-----:|:-------------|
| VQ-VAE | 2017 | NeurIPS | Discrete latent representations |
| SoundStream (RVQ) | 2021 | IEEE | Residual multi-stage quantization |
| Product Quantization | 2011 | IEEE TPAMI | Multi-head independent quantization |
| VQGraph | 2024 | ICLR | GNN-to-MLP distillation via VQ |
| Dr.E | 2024 | — | Codebook-to-vocabulary alignment |
| QGR Survey | 2025 | arXiv | Comprehensive taxonomy of graph quantization |
