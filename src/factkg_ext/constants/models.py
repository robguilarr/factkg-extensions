"""Constants for models and prompts."""

DEFAULT_LLM_MODEL = "gpt-4o"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

PROMPT_TEMPLATE_NO_EVIDENCE = """
Verify the following claim. Respond with only 'SUPPORTED' or 'REFUTED'.
Claim: {claim}
"""

PROMPT_TEMPLATE_WITH_EVIDENCE = """
Verify the following claim based on the provided evidence.
Respond with only 'SUPPORTED' or 'REFUTED'.
Claim: {claim}
Evidence:
{evidence}
"""
