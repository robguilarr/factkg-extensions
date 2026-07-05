# Gap 1: Evaluating Modern LLMs on FactKG

## Motivation

The original FactKG paper evaluates the dataset using a modified GEAR framework and several text-only baselines, including BERT, BlueBERT, and Flan-T5-XL (3B parameters). However, the evaluation of Flan-T5-XL is limited to a zero-shot setting without any retrieved evidence. This leaves a significant gap in understanding how modern, massive Large Language Models (LLMs) perform on complex Knowledge Graph (KG) reasoning tasks, particularly when provided with retrieved evidence in a Retrieval-Augmented Generation (RAG) setup. Evaluating state-of-the-art LLMs on FactKG establishes modern baselines and investigates whether these models have internalized the 2015 DBpedia knowledge or if they still hallucinate on complex multi-hop KG reasoning tasks.

## Mathematical Formulation

The evaluation of LLMs on FactKG can be formulated as a classification task where the model predicts a label $\hat{y} \in \{\text{SUPPORTED}, \text{REFUTED}\}$ for a given claim $c$. In the RAG setup, the model is also provided with retrieved evidence $e$.

Let $f_{\theta}$ be the LLM parameterized by $\theta$. The prediction probability is given by:

$$ P(\hat{y} | c, e; \theta) = f_{\theta}(c, e) $$

The decision rule implemented in the codebase is a simple exact match of the generated text against the ground truth label $y$:

$$ \text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\hat{y}_i = y_i) $$

where $N$ is the total number of claims, and $\mathbb{I}$ is the indicator function.

Furthermore, the accuracy is calculated per reasoning type $t \in \{\text{one-hop}, \text{conjunction}, \text{existence}, \text{multi-hop}, \text{negation}\}$:

$$ \text{Accuracy}_t = \frac{1}{N_t} \sum_{i=1}^{N} \mathbb{I}(\hat{y}_i = y_i \land \text{type}(c_i) = t) $$

where $N_t$ is the number of claims of reasoning type $t$.

## Implementation

The evaluation logic is implemented in the `gap1_llm_rag` module. The core component is the `LLMRAGBenchmarker` class located in `src/factkg_ext/gap1_llm_rag/benchmark.py`.

The `LLMRAGBenchmarker` takes an `LLMClient` instance upon initialization. The `LLMClient` is an abstract interface defined in `src/factkg_ext/common/llm_client.py`, with two concrete implementations:
1.  `MockLLMClient`: A deterministic mock client used for offline testing.
2.  `OpenAILLMClient`: An OpenAI-compatible client that interacts with models like `gpt-4o`.

The `evaluate` method of `LLMRAGBenchmarker` iterates through the provided `claims_data`. For each claim, it constructs a prompt using either `PROMPT_TEMPLATE_NO_EVIDENCE` or `PROMPT_TEMPLATE_WITH_EVIDENCE` (defined in `src/factkg_ext/constants/models.py`), depending on the `use_evidence` flag. It then calls the `generate` method of the `LLMClient` to obtain the prediction.

The method calculates the overall accuracy and the accuracy by reasoning type, returning a `BenchmarkResult` dataclass containing these metrics.

This module serves as a modern alternative to the original `claim_only/flan_xl_zeroshot.py` script, extending the evaluation to support RAG setups and modern LLM APIs.

## Configuration

The configuration for this gap is managed via the `conf/gap1.yaml` file. A typical configuration snippet might look like this:

```yaml
llm_model: "gpt-4o"
gap1_config:
  use_evidence: true
  batch_size: 10
```

*   `llm_model`: Specifies the LLM to be used (e.g., `gpt-4o`).
*   `gap1_config.use_evidence`: A boolean flag indicating whether to include retrieved evidence in the prompt (RAG setup).
*   `gap1_config.batch_size`: The number of claims to process in a single batch (if batching is implemented).

## Step-by-Step Execution

1.  **Clone the original repository (optional for reference):**
    ```bash
    git clone https://github.com/jiho283/FactKG.git
    ```
2.  **Install the extension package:**
    ```bash
    pip install -e ".[dev]"
    ```
3.  **Configure the evaluation:**
    Edit `conf/gap1.yaml` to set the desired `llm_model` and `use_evidence` flag.
4.  **Run the benchmark:**
    ```python
    from factkg_ext.common.llm_client import OpenAILLMClient
    from factkg_ext.gap1_llm_rag.benchmark import LLMRAGBenchmarker
    from factkg_ext.utilities.config import load_config

    config = load_config("conf/gap1.yaml")
    client = OpenAILLMClient(model_name=config.llm_model)
    benchmarker = LLMRAGBenchmarker(client)

    # Load your claims data here
    claims_data = {
        "Paris is in France.": {"Label": ["SUPPORTED"], "Type": "one-hop", "Evidence": "Paris is the capital of France."}
    }

    result = benchmarker.evaluate(claims_data, use_evidence=config.gap1_config.get("use_evidence", False))
    print(f"Overall Accuracy: {result.accuracy}")
    print(f"Accuracy by Type: {result.accuracy_by_type}")
    ```
5.  **Run the tests:**
    ```bash
    pytest tests/test_core.py::test_gap1_benchmark
    ```

## Expected Outcomes & Evaluation

The expected outcome is a comprehensive benchmark of modern LLMs on the FactKG dataset. The results should be compared against the zero-shot Flan-T5-XL baseline reported in the original paper. Specifically, the evaluation should highlight the performance differences across the five reasoning types (one-hop, conjunction, existence, multi-hop, negation) and demonstrate the impact of providing retrieved KG evidence (RAG vs. No Evidence).

Evaluating a subset of the dataset (e.g., the test set of approximately 10k claims) using the OpenAI API (e.g., `gpt-4o`) would require an estimated budget of $50-$200, depending on the prompt length and the specific model chosen.
