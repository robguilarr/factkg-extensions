from typing import List, Tuple

import numpy as np


class DenseSubgraphRetriever:
    """
    Replaces the BERT relation classifier + hop classifier with a bi-encoder dense retriever.
    Scores claim-vs-relation-path similarity.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            self.model = None
            
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        
    def retrieve(
        self, claim: str, candidate_paths: List[str], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Retrieve the most relevant paths for a given claim.
        """
        if not candidate_paths:
            return []
            
        if self.model is None:
            # Fallback for testing without sentence-transformers
            # Simple lexical overlap score
            scores = []
            claim_words = set(claim.lower().split())
            for path in candidate_paths:
                path_words = set(path.lower().split())
                overlap = len(claim_words.intersection(path_words))
                scores.append((path, float(overlap)))
        else:
            # Use dense embeddings
            claim_emb = self.model.encode(claim)
            path_embs = self.model.encode(candidate_paths)
            
            scores = []
            for i, path in enumerate(candidate_paths):
                sim = self._cosine_similarity(claim_emb, path_embs[i])
                scores.append((path, sim))
                
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
