"""Tests for Gap 4: Quantized Evidence Fingerprints.

All tests run offline. Tests validate: fingerprint generation, code alignment,
verification decisions, incremental updates, and temporal stability.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from factkg_ext.gap4_qgr_fingerprint.fingerprinter import (
    ClaimFingerprinter,
    EvidenceFingerprinter,
    EvidenceTriple,
    VQConfig,
    Fingerprint,
)
from factkg_ext.gap4_qgr_fingerprint.verifier import (
    CodeAlignmentVerifier,
    VerificationLabel,
    VerificationResult,
    VerifierConfig,
)


@pytest.fixture
def vq_config():
    return VQConfig(codebook_size=64, code_length=8, embedding_dim=64)


@pytest.fixture
def claim_fp(vq_config):
    return ClaimFingerprinter(vq_config)


@pytest.fixture
def evidence_fp(vq_config):
    return EvidenceFingerprinter(vq_config)


@pytest.fixture
def verifier(claim_fp, evidence_fp):
    config = VerifierConfig(support_threshold=0.5, refute_threshold=0.2)
    return CodeAlignmentVerifier(claim_fp, evidence_fp, config)


@pytest.fixture
def sample_evidence():
    return [
        EvidenceTriple("Barack_Obama", "president_of", "United_States"),
        EvidenceTriple("Barack_Obama", "born_in", "Honolulu"),
        EvidenceTriple("Honolulu", "located_in", "Hawaii"),
    ]


class TestClaimFingerprinter:
    """Tests for claim fingerprinting."""

    def test_fingerprint_produces_codes(self, claim_fp, vq_config):
        """Fingerprint should produce valid codes."""
        fp = claim_fp.fingerprint("Barack Obama was president of the United States")
        assert fp.codes.shape == (vq_config.code_length,)
        assert np.all(fp.codes >= 0)
        assert np.all(fp.codes < vq_config.codebook_size)

    def test_fingerprint_deterministic(self, claim_fp):
        """Same claim should produce same fingerprint."""
        fp1 = claim_fp.fingerprint("Test claim")
        fp2 = claim_fp.fingerprint("Test claim")
        np.testing.assert_array_equal(fp1.codes, fp2.codes)

    def test_different_claims_different_codes(self, claim_fp):
        """Different claims should (usually) produce different codes."""
        fp1 = claim_fp.fingerprint("The sky is blue")
        fp2 = claim_fp.fingerprint("Water boils at 100 degrees Celsius")
        # Not guaranteed to be different for all positions, but should differ somewhere
        assert not np.array_equal(fp1.codes, fp2.codes)

    def test_batch_fingerprint(self, claim_fp):
        """Batch fingerprinting should work."""
        claims = ["Claim A", "Claim B", "Claim C"]
        fps = claim_fp.batch_fingerprint(claims)
        assert len(fps) == 3
        assert all(fp.source_type == "claim" for fp in fps)


class TestEvidenceFingerprinter:
    """Tests for evidence subgraph fingerprinting."""

    def test_fingerprint_produces_codes(self, evidence_fp, sample_evidence, vq_config):
        """Evidence fingerprint should produce valid codes."""
        fp = evidence_fp.fingerprint(sample_evidence)
        assert fp.codes.shape == (vq_config.code_length,)
        assert fp.source_type == "evidence"

    def test_empty_evidence(self, evidence_fp, vq_config):
        """Empty evidence should produce zero embedding fingerprint."""
        fp = evidence_fp.fingerprint([])
        assert fp.codes.shape == (vq_config.code_length,)

    def test_incremental_update(self, evidence_fp, sample_evidence):
        """Incremental update should produce valid fingerprint."""
        original_fp = evidence_fp.fingerprint(sample_evidence)

        # Remove one triple, add another
        removed = [sample_evidence[0]]
        added = [EvidenceTriple("Joe_Biden", "president_of", "United_States")]

        updated_fp = evidence_fp.incremental_update(
            old_fingerprint=original_fp,
            removed_triples=removed,
            added_triples=added,
            total_triples=len(sample_evidence),  # 3 - 1 + 1 = 3
        )
        assert updated_fp.codes.shape == original_fp.codes.shape
        assert updated_fp.source_type == "evidence"

    def test_incremental_vs_full_recompute(self, evidence_fp):
        """Incremental update should approximate full recomputation."""
        triples_v1 = [
            EvidenceTriple("A", "r1", "B"),
            EvidenceTriple("B", "r2", "C"),
            EvidenceTriple("C", "r3", "D"),
        ]
        triples_v2 = [
            EvidenceTriple("B", "r2", "C"),
            EvidenceTriple("C", "r3", "D"),
            EvidenceTriple("D", "r4", "E"),
        ]

        # Full recompute
        fp_full = evidence_fp.fingerprint(triples_v2)

        # Incremental from v1
        fp_v1 = evidence_fp.fingerprint(triples_v1)
        removed = [EvidenceTriple("A", "r1", "B")]
        added = [EvidenceTriple("D", "r4", "E")]
        fp_incr = evidence_fp.incremental_update(fp_v1, removed, added, total_triples=3)

        # Codes should be similar (not necessarily identical due to normalization)
        overlap = np.mean(fp_full.codes == fp_incr.codes)
        assert overlap > 0.3  # At least some overlap expected


class TestCodeAlignmentVerifier:
    """Tests for the verification pipeline."""

    def test_verify_returns_result(self, verifier, sample_evidence):
        """Verification should return a valid result."""
        result = verifier.verify(
            "Barack Obama was born in Honolulu", sample_evidence
        )
        assert isinstance(result.label, VerificationLabel)
        assert 0.0 <= result.alignment_score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_alignment_score_range(self, verifier):
        """Alignment score should be in [0, 1]."""
        codes_a = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        codes_b = np.array([1, 2, 3, 4, 9, 10, 11, 12])
        score = verifier.compute_alignment(codes_a, codes_b)
        assert 0.0 <= score <= 1.0
        assert score == 0.5  # 4 out of 8 match

    def test_identical_codes_perfect_alignment(self, verifier):
        """Identical codes should give alignment = 1.0."""
        codes = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        score = verifier.compute_alignment(codes, codes)
        assert score == 1.0

    def test_no_overlap_zero_alignment(self, verifier):
        """Completely different codes should give alignment = 0.0."""
        codes_a = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        codes_b = np.array([8, 9, 10, 11, 12, 13, 14, 15])
        score = verifier.compute_alignment(codes_a, codes_b)
        assert score == 0.0

    def test_batch_verify(self, verifier, sample_evidence):
        """Batch verification should work."""
        claims = ["Claim 1", "Claim 2"]
        evidence_sets = [sample_evidence, sample_evidence]
        results = verifier.batch_verify(claims, evidence_sets)
        assert len(results) == 2

    def test_temporal_stability(self, verifier):
        """Temporal stability measurement should return valid scores."""
        evidence_before = [
            EvidenceTriple("A", "r", "B"),
            EvidenceTriple("B", "r", "C"),
        ]
        evidence_after = [
            EvidenceTriple("A", "r", "B"),
            EvidenceTriple("B", "r", "D"),  # C changed to D
        ]
        before, after, diff = verifier.compute_temporal_stability(
            "Test claim", evidence_before, evidence_after
        )
        assert 0.0 <= before <= 1.0
        assert 0.0 <= after <= 1.0
        assert diff >= 0.0
