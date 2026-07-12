"""Code Alignment Verifier: Fact verification via discrete code comparison.

Computes alignment score between claim and evidence fingerprints to determine
whether the evidence SUPPORTS, REFUTES, or provides NOT_ENOUGH_INFO for the claim.

The alignment score is based on code overlap (Jaccard similarity) between
the claim fingerprint and the evidence fingerprint. The intuition: if a claim
and its supporting evidence share structural patterns, their quantized codes
will overlap significantly.

Verification decision:
    - SUPPORTS: alignment_score >= support_threshold
    - REFUTES: alignment_score <= refute_threshold
    - NOT_ENOUGH_INFO: otherwise
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from .fingerprinter import Fingerprint, ClaimFingerprinter, EvidenceFingerprinter, VQConfig, EvidenceTriple


class VerificationLabel(Enum):
    """Fact verification labels."""
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


@dataclass
class VerificationResult:
    """Result of fact verification."""

    label: VerificationLabel
    alignment_score: float  # Jaccard similarity between codes
    confidence: float  # How far the score is from the decision boundary
    claim_fingerprint: Fingerprint
    evidence_fingerprint: Fingerprint


@dataclass
class VerifierConfig:
    """Configuration for the code alignment verifier."""

    support_threshold: float = 0.5  # Score above this → SUPPORTS
    refute_threshold: float = 0.2  # Score below this → REFUTES
    use_weighted_alignment: bool = True  # Weight code positions by importance
    position_weights: Optional[np.ndarray] = None  # Custom weights per code position


class CodeAlignmentVerifier:
    """Verifies facts by comparing claim and evidence code fingerprints.

    The verification pipeline:
    1. Receive claim text and evidence triples
    2. Generate fingerprints for both
    3. Compute code alignment score
    4. Apply decision thresholds

    Code alignment score (Jaccard):
        A(claim, evidence) = |{i : claim_codes[i] == evidence_codes[i]}| / L

    Weighted alignment (optional):
        A_w(claim, evidence) = sum(w_i * [claim_codes[i] == evidence_codes[i]]) / sum(w_i)

    where w_i are learned or heuristic weights per code position.
    """

    def __init__(
        self,
        claim_fingerprinter: ClaimFingerprinter,
        evidence_fingerprinter: EvidenceFingerprinter,
        config: Optional[VerifierConfig] = None,
    ):
        self.claim_fp = claim_fingerprinter
        self.evidence_fp = evidence_fingerprinter
        self.config = config or VerifierConfig()

        # Initialize position weights
        if self.config.position_weights is not None:
            self._weights = self.config.position_weights
        else:
            # Default: uniform weights
            code_length = self.claim_fp.config.code_length
            self._weights = np.ones(code_length, dtype=np.float32)

    def compute_alignment(
        self, claim_codes: np.ndarray, evidence_codes: np.ndarray
    ) -> float:
        """Compute alignment score between two code sequences.

        Args:
            claim_codes: Claim fingerprint codes, shape (code_length,).
            evidence_codes: Evidence fingerprint codes, shape (code_length,).

        Returns:
            Alignment score in [0, 1].
        """
        matches = (claim_codes == evidence_codes).astype(np.float32)

        if self.config.use_weighted_alignment:
            score = float(np.sum(matches * self._weights) / np.sum(self._weights))
        else:
            score = float(np.mean(matches))

        return score

    def verify(
        self, claim_text: str, evidence_triples: List[EvidenceTriple]
    ) -> VerificationResult:
        """Verify a claim against evidence triples.

        Args:
            claim_text: The claim to verify.
            evidence_triples: Evidence subgraph triples.

        Returns:
            VerificationResult with label, score, and confidence.
        """
        claim_fp = self.claim_fp.fingerprint(claim_text)
        evidence_fp = self.evidence_fp.fingerprint(evidence_triples)

        alignment = self.compute_alignment(claim_fp.codes, evidence_fp.codes)

        # Decision logic
        if alignment >= self.config.support_threshold:
            label = VerificationLabel.SUPPORTS
            confidence = (alignment - self.config.support_threshold) / (1.0 - self.config.support_threshold)
        elif alignment <= self.config.refute_threshold:
            label = VerificationLabel.REFUTES
            confidence = (self.config.refute_threshold - alignment) / self.config.refute_threshold
        else:
            label = VerificationLabel.NOT_ENOUGH_INFO
            # Confidence is distance to nearest boundary
            dist_to_support = self.config.support_threshold - alignment
            dist_to_refute = alignment - self.config.refute_threshold
            range_width = self.config.support_threshold - self.config.refute_threshold
            confidence = 1.0 - (min(dist_to_support, dist_to_refute) / (range_width / 2))

        confidence = max(0.0, min(1.0, confidence))

        return VerificationResult(
            label=label,
            alignment_score=alignment,
            confidence=confidence,
            claim_fingerprint=claim_fp,
            evidence_fingerprint=evidence_fp,
        )

    def batch_verify(
        self,
        claims: List[str],
        evidence_sets: List[List[EvidenceTriple]],
    ) -> List[VerificationResult]:
        """Verify multiple claims in batch.

        Args:
            claims: List of claim texts.
            evidence_sets: List of evidence triple lists (one per claim).

        Returns:
            List of VerificationResults.
        """
        assert len(claims) == len(evidence_sets)
        return [
            self.verify(claim, evidence)
            for claim, evidence in zip(claims, evidence_sets)
        ]

    def compute_temporal_stability(
        self,
        claim_text: str,
        evidence_before: List[EvidenceTriple],
        evidence_after: List[EvidenceTriple],
    ) -> Tuple[float, float, float]:
        """Measure how stable verification is across KG updates.

        Returns:
            Tuple of (score_before, score_after, score_difference).
        """
        claim_fp = self.claim_fp.fingerprint(claim_text)
        ev_before_fp = self.evidence_fp.fingerprint(evidence_before)
        ev_after_fp = self.evidence_fp.fingerprint(evidence_after)

        score_before = self.compute_alignment(claim_fp.codes, ev_before_fp.codes)
        score_after = self.compute_alignment(claim_fp.codes, ev_after_fp.codes)

        return score_before, score_after, abs(score_after - score_before)
