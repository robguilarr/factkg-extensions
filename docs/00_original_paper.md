# FactKG: Fact Verification via Reasoning on Knowledge Graphs

## 1. Problem Setting and Methodology

The paper "FactKG: Fact Verification via Reasoning on Knowledge Graphs" (Kim et al., ACL 2023) introduces a novel dataset and baseline methodology for fact verification that necessitates reasoning over Knowledge Graphs (KGs). The core problem addressed is the verification of natural language claims against structured knowledge, specifically the DBpedia KG (2015-10 version, containing approximately 0.1 billion triples). The authors generated 108k claims based on graph-text pairs from the WebNLG dataset. These claims are categorized into five distinct reasoning types: One-hop, Conjunction, Existence, Multi-hop, and Negation. Furthermore, the dataset incorporates various linguistic styles, including both colloquial and written forms, to enhance its applicability to real-world scenarios such as dialogue systems.

To evaluate this dataset, the authors proposed a baseline framework adapted from GEAR (Graph-based Evidence Aggregating and Reasoning). The pipeline is composed of two primary modules: the Subgraph Retrieval Module and the Claim Verification Module.

The **Subgraph Retrieval Module** replaces the traditional document retrieval and sentence selection components found in the original GEAR framework. It utilizes two independent BERT models:
1.  **Relation Classifier**: Predicts the set of relations relevant to the claim and the entity.
2.  **Hop Classifier**: Predicts the maximum number of hops to be traversed from the entity within the KG.

Based on the predictions from these classifiers, the module extracts a subgraph from the KG, identifying paths that connect the entities mentioned in the claim.

The **Claim Verification Module** employs the claim verification component from GEAR. It uses a Sentence Encoder (BERT) to extract the final hidden state of both the claim and the retrieved graph evidence. An Evidence Aggregator, implemented as a Graph Attention Network (GAT), then aggregates the information from the evidence paths. Finally, a Label Predictor, utilizing a Multi-Layer Perceptron (MLP), classifies the claim into one of two categories: `SUPPORTED` or `REFUTED`.

## 2. Datasets and Key Quantitative Results

The experiments were conducted on the proposed FactKG dataset, which consists of 108k claims divided into train, development, and test sets with an 8:1:1 ratio. The evidence KG utilized is the entire DBpedia (2015-10 version).

The authors evaluated their approach under two main settings:
1.  **Claim Only**: This setting utilized BERT (bert-base-uncased), BlueBERT, and Flan-T5 (google/flan-t5-xl, 3B parameters) without any retrieved evidence.
2.  **With Graphical Evidence**: This setting employed the modified GEAR framework, utilizing BERT for subgraph retrieval (Relation Classifier and Hop Classifier) and claim verification (Sentence Encoder).

The primary evaluation metric was Accuracy in predicting `SUPPORTED` or `REFUTED`. The key quantitative results demonstrated that incorporating graphical evidence significantly improved performance compared to the "Claim Only" baselines. The modified GEAR model with graphical evidence outperformed the text-only baselines, underscoring the necessity of KG reasoning for the FactKG dataset. Additionally, cross-style evaluation revealed that training on colloquial style claims and testing on written style claims sometimes improved performance, indicating the generalization benefits of incorporating colloquial styles.

## 3. Original Codebase Organization

The original FactKG repository (https://github.com/jiho283/FactKG) is organized to reflect the pipeline described above. The key components and their corresponding files are as follows:

*   **Claim Only Baselines**: Implemented in `claim_only/bert_classification.py` and `claim_only/flan_xl_zeroshot.py`.
*   **Subgraph Retrieval Module**:
    *   **Relation Classifier**: Implemented in `with_evidence/retrieve/model/relation_predict/`. The core model is `FactKGRelationClassifier(pl.LightningModule)(relations, model, top_k, learning_rate)`.
    *   **Hop Classifier**: Implemented in `with_evidence/retrieve/model/hop_predict/`.
*   **Claim Verification Module**: The GEAR-style GAT verifier is implemented in `with_evidence/classifier/baseline.py`.
*   **Configuration**: YAML configuration files such as `relation_predict_top3.yaml`, `relation_predict_top5.yaml`, and `relation_predict_top10.yaml` are used to manage hyperparameters.
*   **Data**: The repository utilizes data pickles, including a claims dictionary with the structure `{claim: {'Label', 'Entity_set', 'Evidence'}}` and `dbpedia_2015_undirected_light.pickle`.

## 4. PoC Repository Adapter Layer

This Proof-of-Concept (PoC) repository (`factkg-extensions`) is designed to extend and improve upon the original FactKG codebase. To ensure compatibility and facilitate a smooth transition, it includes an adapter layer that mirrors the original hooks.

The adapter layer is located in `src/factkg_ext/original/adapters.py` and includes the following classes:

*   `FactKGRelationClassifierAdapter`: This class wraps the original `FactKGRelationClassifier` found in `with_evidence/retrieve/model/relation_predict/model.py`. It provides a `predict(claim: str, entity: str) -> List[str]` method that stubs the original prediction logic, returning the top-k relations.
*   `FactKGHopClassifierAdapter`: This class wraps the original `HopPredictorManager` found in `with_evidence/retrieve/model/hop_predict/model.py`. It provides a `predict(claim: str, entity: str) -> int` method that stubs the original prediction logic, returning the predicted number of hops.
*   `FactKGVerifierAdapter`: This class wraps the original GEAR-style GAT verifier found in `with_evidence/classifier/baseline.py`. It provides a `verify(claim: str, evidence_graph: Any) -> str` method that stubs the original verification logic, returning a `SUPPORTED` or `REFUTED` label.

These adapters allow the new extension modules (Gap 1, Gap 2, and Gap 3) to interface seamlessly with the existing pipeline structure, enabling researchers to easily swap out components and evaluate new methodologies against the original baselines.
