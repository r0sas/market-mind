from typing import Optional
from .client import UnifiedAIClient
from .enums import AIProvider

class AIInsightsGeneratorBase:
    """Compatibility base class for existing AI modules."""

    def __init__(self, api_key: Optional[str] = None, use_ollama: bool = False, model: Optional[str] = None):
        if use_ollama:
            provider = AIProvider.OLLAMA
        elif api_key:
            provider = AIProvider.GROQ
        else:
            provider = AIProvider.NONE

        self.ai_client = UnifiedAIClient(provider=provider, api_key=api_key, model=model)
        self.api_key = api_key
        self.use_ollama = use_ollama

    def test_connection(self) -> bool:
        return self.ai_client.test_connection()
