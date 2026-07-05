import yaml

from factkg_ext.common.llm_client import MockLLMClient
from factkg_ext.gap1_llm_rag.benchmark import LLMRAGBenchmarker
from factkg_ext.gap2_kg_update.pipeline import KGVersionUpdater
from factkg_ext.gap3_dense_retrieval.retriever import DenseSubgraphRetriever
from factkg_ext.original.adapters import (
    FactKGHopClassifierAdapter,
    FactKGRelationClassifierAdapter,
    FactKGVerifierAdapter,
)
from factkg_ext.utilities.config import load_config


def test_mock_llm_client():
    client = MockLLMClient(responses={"Paris": "SUPPORTED", "Mars": "REFUTED"})
    assert client.generate("Is Paris in France?") == "SUPPORTED"
    assert client.generate("Is Mars a star?") == "REFUTED"
    assert client.generate("Unknown claim") == "SUPPORTED" # Default

def test_gap1_benchmark():
    client = MockLLMClient(responses={
        "Paris": "SUPPORTED",
        "Mars": "REFUTED"
    })
    benchmarker = LLMRAGBenchmarker(client)
    
    claims_data = {
        "Paris is in France.": {"Label": ["SUPPORTED"], "Type": "one-hop"},
        "Mars is a star.": {"Label": ["REFUTED"], "Type": "one-hop"},
        "London is in Germany.": {"Label": ["REFUTED"], "Type": "multi-hop"}
    }
    
    result = benchmarker.evaluate(claims_data, use_evidence=False)
    assert result.total_claims == 3
    assert result.accuracy == 2/3
    assert result.accuracy_by_type["one-hop"] == 1.0
    assert result.accuracy_by_type["multi-hop"] == 0.0

def test_gap2_kg_update():
    updater = KGVersionUpdater()
    
    claims_data = {
        "Valid claim": {"Evidence_Triples": [("ValidSubject", "is", "ValidObject")]},
        "Stale claim": {"Evidence_Triples": [("StaleSubject", "is", "StaleObject")]}
    }
    
    result = updater.update_dataset(claims_data)
    assert len(result.stale_claims) == 1
    assert "Stale claim" in result.stale_claims
    assert len(result.updated_claims) == 1
    assert "Valid claim" in result.updated_claims

def test_gap3_dense_retrieval():
    retriever = DenseSubgraphRetriever()
    
    claim = "Paris is the capital of France"
    candidate_paths = [
        "Paris capital France",
        "London capital UK",
        "Paris city"
    ]
    
    results = retriever.retrieve(claim, candidate_paths, top_k=2)
    assert len(results) == 2
    # Lexical fallback should score "Paris capital France" highest
    assert results[0][0] == "Paris capital France"

def test_config_loader(tmp_path):
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump({"llm_model": "test-model", "gap1_config": {"param": 1}}, f)
        
    config = load_config(str(config_path))
    assert config.llm_model == "test-model"
    assert config.gap1_config["param"] == 1

def test_adapters():
    rel_adapter = FactKGRelationClassifierAdapter(relations=["r1", "r2", "r3"], top_k=2)
    assert len(rel_adapter.predict("claim", "entity")) == 2
    
    hop_adapter = FactKGHopClassifierAdapter()
    assert hop_adapter.predict("claim", "entity") == 1
    
    verifier = FactKGVerifierAdapter()
    assert verifier.verify("claim", None) == "SUPPORTED"
