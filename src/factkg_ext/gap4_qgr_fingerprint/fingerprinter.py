"""Claim and Evidence Fingerprinters: Encode text/graphs into discrete code fingerprints.

This module provides two fingerprinting classes:
1. ClaimFingerprinter: Encodes claim text into a dense embedding, then quantizes
   it into a discrete code fingerprint.
2. EvidenceFingerprinter: Encodes evidence subgraph structure (triples) into a
   dense embedding, then quantizes it into a discrete code fingerprint.

Both use the same VQ codebook infrastructure, enabling direct code comparison
for fact verification.

Key advantage over continuous embeddings: When the KG updates, only the affected
entity codes need re-quantization (O(affected entities)), not full re-embedding
of the entire graph (O(all entities)).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class VQConfig:
    """Configuration for the fingerprint VQ codebook."""

    codebook_size: int = 512
    code_length: int = 16
    embedding_dim: int = 256
    commitment_beta: float = 0.25
    ema_decay: float = 0.99
    epsilon: float = 1e-5


@dataclass
class Fingerprint:
    """A discrete code fingerprint."""

    codes: np.ndarray  # Shape: (code_length,)
    embedding: np.ndarray  # Original dense embedding before quantization
    reconstruction_mse: float
    source_type: str  # "claim" or "evidence"


@dataclass
class EvidenceTriple:
    """A triple from the evidence subgraph."""

    subject: str
    predicate: str
    obj: str


class _VQCodebook:
    """Shared VQ codebook implementation for fingerprinting.

    Multi-head VQ: embedding is split into code_length chunks,
    each quantized independently.
    """

    def __init__(self, config: VQConfig):
        self.config = config
        self.chunk_dim = config.embedding_dim // config.code_length
        assert config.embedding_dim % config.code_length == 0

        scale = (2.0 / (self.chunk_dim + config.codebook_size)) ** 0.5
        self.codebooks = np.random.randn(
            config.code_length, config.codebook_size, self.chunk_dim
        ).astype(np.float32) * scale

        self.ema_count = np.ones(
            (config.code_length, config.codebook_size), dtype=np.float32
        )
        self.ema_sum = self.codebooks.copy()
        self._training = True

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode embedding to codes. x: (embedding_dim,) -> (code_length,)"""
        if x.ndim == 1:
            x = x[np.newaxis, :]
        batch = x.shape[0]
        chunks = x.reshape(batch, self.config.code_length, self.chunk_dim)
        codes = np.zeros((batch, self.config.code_length), dtype=np.int64)

        for h in range(self.config.code_length):
            chunk = chunks[:, h, :]
            cb = self.codebooks[h]
            dists = (
                np.sum(chunk ** 2, axis=1, keepdims=True)
                - 2 * chunk @ cb.T
                + np.sum(cb ** 2, axis=1, keepdims=True).T
            )
            codes[:, h] = np.argmin(dists, axis=1)

            if self._training:
                self._ema_update(h, codes[:, h], chunk)

        return codes[0] if batch == 1 else codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode codes to embedding. codes: (code_length,) -> (embedding_dim,)"""
        if codes.ndim == 1:
            codes = codes[np.newaxis, :]
        batch = codes.shape[0]
        chunks = np.zeros((batch, self.config.code_length, self.chunk_dim), dtype=np.float32)
        for h in range(self.config.code_length):
            chunks[:, h, :] = self.codebooks[h][codes[:, h]]
        result = chunks.reshape(batch, self.config.embedding_dim)
        return result[0] if batch == 1 else result

    def _ema_update(self, head: int, indices: np.ndarray, vectors: np.ndarray) -> None:
        gamma = self.config.ema_decay
        eps = self.config.epsilon
        counts = np.bincount(indices, minlength=self.config.codebook_size).astype(np.float32)
        sums = np.zeros_like(self.codebooks[head])
        for j in range(self.config.codebook_size):
            mask = indices == j
            if np.any(mask):
                sums[j] = vectors[mask].sum(axis=0)
        self.ema_count[head] = gamma * self.ema_count[head] + (1 - gamma) * counts
        self.ema_sum[head] = gamma * self.ema_sum[head] + (1 - gamma) * sums
        n = self.ema_count[head].sum()
        smoothed = (self.ema_count[head] + eps) / (n + self.config.codebook_size * eps) * n
        self.codebooks[head] = self.ema_sum[head] / smoothed[:, np.newaxis]

    def compute_perplexity(self, codes: np.ndarray) -> float:
        """Compute average codebook perplexity across heads."""
        if codes.ndim == 1:
            codes = codes[np.newaxis, :]
        perplexities = []
        for h in range(self.config.code_length):
            counts = np.bincount(codes[:, h], minlength=self.config.codebook_size).astype(np.float32)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs))
            perplexities.append(float(np.exp(entropy)))
        return float(np.mean(perplexities))


class ClaimFingerprinter:
    """Encodes claim text into a discrete code fingerprint.

    The encoding process:
    1. Convert claim text to a dense embedding (simulated via hashing for offline use,
       or via a sentence transformer in production).
    2. Quantize the embedding using multi-head VQ.
    3. Return the code fingerprint.

    In production, step 1 would use a sentence-transformers model like
    all-MiniLM-L6-v2 or a fine-tuned claim encoder.
    """

    def __init__(self, config: Optional[VQConfig] = None):
        self.config = config or VQConfig()
        self.codebook = _VQCodebook(self.config)

    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Convert text to dense embedding.

        For offline/testing: uses deterministic hash-based embedding.
        For production: replace with sentence-transformers encode().
        """
        seed = hash(text) % (2**31)
        rng = np.random.RandomState(seed)
        embedding = rng.randn(self.config.embedding_dim).astype(np.float32)
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        return embedding

    def fingerprint(self, claim_text: str) -> Fingerprint:
        """Create a code fingerprint for a claim.

        Args:
            claim_text: The claim text to fingerprint.

        Returns:
            Fingerprint with discrete codes and metadata.
        """
        embedding = self._text_to_embedding(claim_text)
        codes = self.codebook.encode(embedding)
        reconstructed = self.codebook.decode(codes)
        mse = float(np.mean((embedding - reconstructed) ** 2))

        return Fingerprint(
            codes=codes,
            embedding=embedding,
            reconstruction_mse=mse,
            source_type="claim",
        )

    def batch_fingerprint(self, claims: List[str]) -> List[Fingerprint]:
        """Fingerprint multiple claims."""
        return [self.fingerprint(c) for c in claims]


class EvidenceFingerprinter:
    """Encodes evidence subgraph structure into a discrete code fingerprint.

    The encoding process:
    1. Encode each triple in the evidence subgraph as: hash(s) + hash(p) + hash(o).
    2. Aggregate triple embeddings via mean pooling.
    3. Quantize the aggregated embedding using multi-head VQ.

    The structural encoding captures:
    - Entity co-occurrence patterns
    - Relation type distribution
    - Subgraph connectivity (via aggregation)

    Incremental update: When a triple changes, only its contribution to the
    aggregate needs recomputation. If the entity set is small relative to the
    full KG, this is much cheaper than full re-embedding.
    """

    def __init__(self, config: Optional[VQConfig] = None):
        self.config = config or VQConfig()
        self.codebook = _VQCodebook(self.config)
        self._entity_cache: Dict[str, np.ndarray] = {}

    def _entity_to_embedding(self, entity: str) -> np.ndarray:
        """Get or create entity embedding."""
        if entity not in self._entity_cache:
            seed = hash(entity) % (2**31)
            rng = np.random.RandomState(seed)
            self._entity_cache[entity] = rng.randn(self.config.embedding_dim).astype(np.float32)
        return self._entity_cache[entity]

    def _encode_triple(self, triple: EvidenceTriple) -> np.ndarray:
        """Encode a single triple into a dense vector."""
        s = self._entity_to_embedding(triple.subject)
        p = self._entity_to_embedding(triple.predicate)
        o = self._entity_to_embedding(triple.obj)
        combined = s + p + o
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined /= norm
        return combined

    def fingerprint(self, evidence_triples: List[EvidenceTriple]) -> Fingerprint:
        """Create a code fingerprint for an evidence subgraph.

        Args:
            evidence_triples: List of triples forming the evidence.

        Returns:
            Fingerprint with discrete codes and metadata.
        """
        if not evidence_triples:
            embedding = np.zeros(self.config.embedding_dim, dtype=np.float32)
        else:
            triple_embeddings = np.stack([self._encode_triple(t) for t in evidence_triples])
            embedding = triple_embeddings.mean(axis=0)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding /= norm

        codes = self.codebook.encode(embedding)
        reconstructed = self.codebook.decode(codes)
        mse = float(np.mean((embedding - reconstructed) ** 2))

        return Fingerprint(
            codes=codes,
            embedding=embedding,
            reconstruction_mse=mse,
            source_type="evidence",
        )

    def incremental_update(
        self,
        old_fingerprint: Fingerprint,
        removed_triples: List[EvidenceTriple],
        added_triples: List[EvidenceTriple],
        total_triples: int,
    ) -> Fingerprint:
        """Incrementally update a fingerprint when the KG changes.

        Instead of re-encoding the entire evidence subgraph, subtract removed
        triple contributions and add new ones.

        Args:
            old_fingerprint: The existing fingerprint to update.
            removed_triples: Triples removed from the evidence.
            added_triples: Triples added to the evidence.
            total_triples: Total number of triples after update.

        Returns:
            Updated Fingerprint.
        """
        # Reconstruct the aggregate embedding
        old_embedding = old_fingerprint.embedding.copy()
        old_total = total_triples + len(removed_triples) - len(added_triples)

        if old_total > 0:
            # Un-normalize and scale back to sum
            aggregate = old_embedding * old_total
        else:
            aggregate = np.zeros(self.config.embedding_dim, dtype=np.float32)

        # Subtract removed
        for t in removed_triples:
            aggregate -= self._encode_triple(t)

        # Add new
        for t in added_triples:
            aggregate += self._encode_triple(t)

        # Re-normalize
        if total_triples > 0:
            new_embedding = aggregate / total_triples
            norm = np.linalg.norm(new_embedding)
            if norm > 0:
                new_embedding /= norm
        else:
            new_embedding = np.zeros(self.config.embedding_dim, dtype=np.float32)

        codes = self.codebook.encode(new_embedding)
        reconstructed = self.codebook.decode(codes)
        mse = float(np.mean((new_embedding - reconstructed) ** 2))

        return Fingerprint(
            codes=codes,
            embedding=new_embedding,
            reconstruction_mse=mse,
            source_type="evidence",
        )
