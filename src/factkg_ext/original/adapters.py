"""
Adapters for the original FactKG codebase.
These interfaces show how new code hooks into the original codebase.
"""
from typing import Any, List


class FactKGRelationClassifierAdapter:
    """
    Wraps the original FactKGRelationClassifier.
    Original file: with_evidence/retrieve/model/relation_predict/model.py
    """
    def __init__(self, relations: List[str], top_k: int = 5):
        self.relations = relations
        self.top_k = top_k
        
    def predict(self, claim: str, entity: str) -> List[str]:
        """Predict relations for a given claim and entity."""
        # Stub for the original prediction logic
        return self.relations[:self.top_k]

class FactKGHopClassifierAdapter:
    """
    Wraps the original HopPredictorManager.
    Original file: with_evidence/retrieve/model/hop_predict/model.py
    """
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        
    def predict(self, claim: str, entity: str) -> int:
        """Predict the number of hops for a given claim and entity."""
        # Stub for the original prediction logic
        return 1

class FactKGVerifierAdapter:
    """
    Wraps the original GEAR-style GAT verifier.
    Original file: with_evidence/classifier/baseline.py
    """
    def __init__(self):
        pass
        
    def verify(self, claim: str, evidence_graph: Any) -> str:
        """Verify a claim given an evidence graph."""
        # Stub for the original verification logic
        return "SUPPORTED"
