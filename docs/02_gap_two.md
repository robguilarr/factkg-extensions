# Gap 2: Updating FactKG to Recent DBpedia/Wikidata Versions

## Motivation

A significant limitation explicitly stated by the authors of the original FactKG paper is the reliance on outdated knowledge. The dataset is derived from the 2015-10 version of DBpedia, meaning it does not reflect the latest real-world facts. This gap addresses this limitation by providing a pipeline to remap existing claims and entities to current DBpedia or Wikidata releases. By filtering out claims whose truth values have changed or flagging them as stale, this extension introduces a temporal dimension to KG-based fact verification, making the benchmark highly relevant for real-world applications where knowledge is constantly evolving.

## Mathematical Formulation

The objective is to determine the validity of a claim $c$ based on its supporting evidence triples $E_c = \{(s_i, p_i, o_i)\}_{i=1}^{k}$ against a current Knowledge Graph $\mathcal{G}_{current}$.

Let $V(s, p, o, \mathcal{G})$ be a verification function that returns $1$ if the triple $(s, p, o)$ exists in the graph $\mathcal{G}$, and $0$ otherwise.

The decision rule for flagging a claim as "stale" is defined as:

$$ \text{IsStale}(c) = \begin{cases} 1, & \text{if } \exists (s, p, o) \in E_c \text{ such that } V(s, p, o, \mathcal{G}_{current}) = 0 \\ 0, & \text{otherwise} \end{cases} $$

In the implemented codebase, the verification function $V$ is approximated using a SPARQL `ASK` query against a specified endpoint. The objective is to partition the original dataset $D$ into an updated dataset $D_{updated}$ and a set of stale claims $D_{stale}$:

$$ D_{updated} = \{ c \in D \mid \text{IsStale}(c) = 0 \} $$
$$ D_{stale} = \{ c \in D \mid \text{IsStale}(c) = 1 \} $$

## Implementation

The logic for updating the dataset is implemented in the `gap2_kg_update` module. The core component is the `KGVersionUpdater` class located in `src/factkg_ext/gap2_kg_update/pipeline.py`.

The `KGVersionUpdater` is initialized with a SPARQL `endpoint_url` (defaulting to `https://dbpedia.org/sparql`). The primary method, `update_dataset`, takes a dictionary of `claims_data`. For each claim, it extracts the associated `Evidence_Triples`.

It then iterates through these triples and calls the `check_fact_exists` method. In a full production environment, this method executes a SPARQL `ASK` query against the configured endpoint to verify the existence of the triple $(s, p, o)$. For the purposes of this Proof-of-Concept (PoC), the method simulates the result by checking if the string "stale" is present in the subject's name.

If any triple associated with a claim is not found in the current KG, the claim is appended to the `stale_claims` list. Otherwise, it is added to the `updated_claims` dictionary. The method returns an `UpdateResult` dataclass containing both collections.

This module provides a crucial data preprocessing step that was absent in the original repository, allowing researchers to adapt the static `dbpedia_2015_undirected_light.pickle` data to modern KG snapshots.

## Configuration

The configuration for this gap is managed via the `conf/gap2.yaml` file. A typical configuration snippet might look like this:

```yaml
gap2_config:
  endpoint_url: "https://dbpedia.org/sparql"
  batch_size: 100
```

*   `gap2_config.endpoint_url`: The URL of the SPARQL endpoint to query (e.g., DBpedia or a local Wikidata instance).
*   `gap2_config.batch_size`: The number of queries to batch together (if batching is implemented to avoid rate limits).

## Step-by-Step Execution

1.  **Clone the original repository (optional for reference):**
    ```bash
    git clone https://github.com/jiho283/FactKG.git
    ```
2.  **Install the extension package:**
    ```bash
    pip install -e ".[dev]"
    ```
3.  **Configure the pipeline:**
    Edit `conf/gap2.yaml` to set the desired `endpoint_url`.
4.  **Run the update pipeline:**
    ```python
    from factkg_ext.gap2_kg_update.pipeline import KGVersionUpdater
    from factkg_ext.utilities.config import load_config

    config = load_config("conf/gap2.yaml")
    endpoint = config.gap2_config.get("endpoint_url", "https://dbpedia.org/sparql") if config.gap2_config else "https://dbpedia.org/sparql"
    updater = KGVersionUpdater(endpoint_url=endpoint)

    # Load your claims data here
    claims_data = {
        "Valid claim": {"Evidence_Triples": [("ValidSubject", "is", "ValidObject")]},
        "Stale claim": {"Evidence_Triples": [("StaleSubject", "is", "StaleObject")]}
    }

    result = updater.update_dataset(claims_data)
    print(f"Updated Claims: {len(result.updated_claims)}")
    print(f"Stale Claims: {len(result.stale_claims)}")
    ```
5.  **Run the tests:**
    ```bash
    pytest tests/test_core.py::test_gap2_kg_update
    ```

## Expected Outcomes & Evaluation

The expected outcome is a partitioned dataset where claims are separated into those that remain valid in the current KG and those that have become stale. This allows for the creation of a temporal fact verification benchmark. The evaluation of this module itself is based on the successful identification of stale facts. When evaluating models on the updated dataset, the metrics (Accuracy) should be compared against the baselines established on the original 2015 dataset to understand how temporal shifts affect model performance. The compute required for this data processing step is minimal, relying primarily on CPU and network requests to the SPARQL endpoint, with an estimated budget of <$50.
