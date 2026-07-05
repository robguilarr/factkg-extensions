from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMClient(ABC):
    """Abstract interface for LLM interactions."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from the LLM."""
        pass

class MockLLMClient(LLMClient):
    """Deterministic mock LLM client for offline testing."""
    
    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.calls: List[str] = []
        
    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append(prompt)
        # Return a predefined response if available, else a default
        for key, value in self.responses.items():
            if key in prompt:
                return value
        return "SUPPORTED" # Default mock response

class OpenAILLMClient(LLMClient):
    """OpenAI-compatible LLM client."""
    
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        self.model_name = model_name
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            self.client = None
            
    def generate(self, prompt: str, **kwargs: Any) -> str:
        if self.client is None:
            raise ImportError("openai package is required for OpenAILLMClient")
            
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content or ""
