# Gap 3: Improving Subgraph Retrieval with Dense Retrieval Methods

## Motivation

The original FactKG baseline employs a rigid Subgraph Retrieval Module consisting of a Relation Classifier and a Hop Classifier. This approach predicts the exact relations and the maximum number of hops to traverse from the entity mentioned in the claim. While functional, this method can be brittle, especially when dealing with colloquial claims where the linguistic expression of a relation might not perfectly align with the KG's schema. Dense retrieval methods, such as those using bi-encoders (e.g., DPR or ColBERT), have proven highly effective in text-based Question Answering (QA). By embedding both the claim and the candidate KG paths into a shared dense vector space, we can retrieve the most relevant paths based on semantic similarity, potentially significantly improving the recall of relevant evidence paths compared to the original classification-based approach.

## Mathematical Formulation

The goal of the dense retriever is to score the relevance of a candidate KG path $p$ given a claim $c$.

Let $E(\cdot)$ be a dense encoder function (e.g., a SentenceTransformer model) that maps a text string to a $d$-dimensional real-valued vector: $E: \text{String} \rightarrow \mathbb{R}^d$.

The claim embedding is $v_c = E(c)$ and the path embedding is $v_p = E(p)$.

The relevance score $S(c, p)$ is computed using the cosine similarity between the two embeddings:

$$ S(c, p) = \frac{v_c \cdot v_p}{\|v_c\| \|v_p\|} $$

Given a set of candidate paths $P = \{p_1, p_2, \dots, p_m\}$, the retriever selects the top-$k$ paths that maximize this similarity score:

$$ P_{top\_k} = \underset{P' \subset P, |P'|=k}{\text{argmax}} \sum_{p \in P'} S(c, p) $$

In the absence of the dense encoder (e.g., during fallback testing), a simple lexical overlap score is used:

$$ S_{lexical}(c, p) = | \text{words}(c) \cap \text{words}(p) | $$

## Implementation

The dense retrieval logic is implemented in the `gap3_dense_retrieval` module. The core component is the `DenseSubgraphRetriever` class located in `src/factkg_ext/gap3_dense_retrieval/retriever.py`.

This class is designed to replace the original `FactKGRelationClassifier` and `HopPredictorManager` (which are stubbed in `src/factkg_ext/original/adapters.py`).

The `DenseSubgraphRetriever` is initialized with a `model_name` (defaulting to `all-MiniLM-L6-v2`). It attempts to load this model using the `sentence_transformers` library.

The primary method, `retrieve`, takes a `claim`, a list of `candidate_paths`, and a `top_k` parameter.
1.  If the `sentence_transformers` model is successfully loaded, it encodes the claim and all candidate paths into dense vectors. It then calculates the cosine similarity (using the internal `_cosine_similarity` method) between the claim vector and each path vector.
2.  If the model fails to load (e.g., due to missing dependencies), it falls back to a simple lexical overlap scoring mechanism for testing purposes.

The method sorts the candidate paths based on their computed scores in descending order and returns the top-$k$ paths along with their scores as a list of tuples: `List[Tuple[str, float]]`.

## Configuration

The configuration for this gap is managed via the `conf/gap3.yaml` file. A typical configuration snippet might look like this:

```yaml
embedding_model: "all-MiniLM-L6-v2"
gap3_config:
  top_k: 5
```

*   `embedding_model`: Specifies the pre-trained SentenceTransformer model to use for encoding (e.g., `all-MiniLM-L6-v2`).
*   `gap3_config.top_k`: The number of top relevant paths to retrieve.

## Step-by-Step Execution

1.  **Clone the original repository (optional for reference):**
    ```bash
    git clone https://github.com/jiho283/FactKG.git
    ```
2.  **Install the extension package (ensure sentence-transformers is installed):**
    ```bash
    pip install -e ".[dev]"
    pip install sentence-transformers
    ```
3.  **Configure the retriever:**
    Edit `conf/gap3.yaml` to set the desired `embedding_model` and `top_k`.
4.  **Run the retriever:**
    ```python
    from factkg_ext.gap3_dense_retrieval.retriever import DenseSubgraphRetriever
    from factkg_ext.utilities.config import load_config

    config = load_config("conf/gap3.yaml")
    retriever = DenseSubgraphRetriever(model_name=config.embedding_model)

    claim = "Paris is the capital of France"
    candidate_paths = [
        "Paris capital France",
        "London capital UK",
        "Paris city"
    ]
    top_k = config.gap3_config.get("top_k", 5) if config.gap3_config else 5

    results = retriever.retrieve(claim, candidate_paths, top_k=top_k)
    for path, score in results:
        print(f"Path: {path}, Score: {score:.4f}")
    ```
5.  **Run the tests:**
    ```bash
    pytest tests/test_core.py::test_gap3_dense_retrieval
    ```

## Expected Outcomes & Evaluation

The expected outcome is a more robust and semantically aware subgraph retrieval process. The evaluation should focus on the recall of the retrieved evidence paths compared to the ground truth evidence provided in the FactKG dataset. This dense retrieval approach should be compared against the original baseline's Relation + Hop classifier approach. We expect to see improved recall, particularly for claims written in a colloquial style, which should translate to higher downstream verification accuracy. Fine-tuning the dense retriever (e.g., a BERT-based model) on the FactKG training set would require a single GPU (e.g., RTX 3090/4090) and approximately 1-3 days of training time.
