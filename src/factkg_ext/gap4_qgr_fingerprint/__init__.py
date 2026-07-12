"""Gap 4: Quantized Evidence Fingerprints for Temporal-Robust Fact Verification.

Implements discrete code fingerprints for claims and evidence subgraphs,
enabling verification via code alignment and incremental KG updates.
"""

from .fingerprinter import ClaimFingerprinter, EvidenceFingerprinter, VQConfig
from .verifier import CodeAlignmentVerifier, VerificationResult

__all__ = [
    "ClaimFingerprinter",
    "EvidenceFingerprinter",
    "VQConfig",
    "CodeAlignmentVerifier",
    "VerificationResult",
]
