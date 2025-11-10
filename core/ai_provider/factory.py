from typing import Optional
from .client import UnifiedAIClient
from .enums import AIProvider

def create_ai_client(provider_name: str, api_key: Optional[str] = None, model: Optional[str] = None) -> UnifiedAIClient:
    provider_map = {
        "groq": AIProvider.GROQ,
        "ollama": AIProvider.OLLAMA,
        "none": AIProvider.NONE,
        "manual": AIProvider.NONE
    }
    provider = provider_map.get(provider_name.lower(), AIProvider.NONE)
    return UnifiedAIClient(provider=provider, api_key=api_key, model=model)
