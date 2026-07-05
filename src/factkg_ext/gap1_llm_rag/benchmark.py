from dataclasses import dataclass
from typing import Any, Dict

from factkg_ext.common.llm_client import LLMClient
from factkg_ext.constants.models import PROMPT_TEMPLATE_NO_EVIDENCE, PROMPT_TEMPLATE_WITH_EVIDENCE


@dataclass
class BenchmarkResult:
    accuracy: float
    accuracy_by_type: Dict[str, float]
    total_claims: int

class LLMRAGBenchmarker:
    """Evaluates LLMs on FactKG claims with and without retrieved KG evidence."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        
    def evaluate(
        self, claims_data: Dict[str, Dict[str, Any]], use_evidence: bool = False
    ) -> BenchmarkResult:
        """
        Evaluate the LLM on a set of claims.
        claims_data format: {claim_text: {'Label': ['SUPPORTED'], 'Type': 'one-hop'}}
        """
        correct = 0
        total = len(claims_data)
        type_correct: Dict[str, int] = {}
        type_total: Dict[str, int] = {}
        
        for claim, data in claims_data.items():
            label = data.get('Label', [''])[0]
            evidence = data.get('Evidence', '')
            reasoning_type = data.get('Type', 'unknown')
            
            if use_evidence and evidence:
                prompt = PROMPT_TEMPLATE_WITH_EVIDENCE.format(claim=claim, evidence=evidence)
            else:
                prompt = PROMPT_TEMPLATE_NO_EVIDENCE.format(claim=claim)
                
            prediction = self.llm_client.generate(prompt).strip().upper()
            
            # Simple parsing of prediction
            is_correct = (prediction == label.upper())
            
            if is_correct:
                correct += 1
                type_correct[reasoning_type] = type_correct.get(reasoning_type, 0) + 1
                
            type_total[reasoning_type] = type_total.get(reasoning_type, 0) + 1
            
        accuracy = correct / total if total > 0 else 0.0
        accuracy_by_type = {
            rtype: (type_correct.get(rtype, 0) / type_total[rtype])
            for rtype in type_total
        }
        
        return BenchmarkResult(
            accuracy=accuracy,
            accuracy_by_type=accuracy_by_type,
            total_claims=total
        )
