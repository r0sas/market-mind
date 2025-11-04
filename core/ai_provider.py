# core/ai_provider.py
"""
Unified AI Provider - Switch between Groq API and Ollama (Local Llama3)
Use this everywhere in the app for consistent AI provider management
"""

import logging
from typing import Optional, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class AIProvider(Enum):
    """Available AI providers"""
    GROQ = "groq"
    OLLAMA = "ollama"
    NONE = "none"


class UnifiedAIClient:
    """
    Unified interface for AI calls - works with both Groq and Ollama.
    
    Usage:
        client = UnifiedAIClient(provider=AIProvider.OLLAMA)
        response = client.chat(messages=[...], temperature=0.3)
    """
    
    def __init__(
        self,
        provider: AIProvider = AIProvider.NONE,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize AI client.
        
        Args:
            provider: Which AI provider to use (GROQ, OLLAMA, or NONE)
            api_key: API key (only needed for GROQ)
            model: Model name (optional, uses defaults)
        """
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.client = None
        self.is_available = False
        
        # Default models
        self.default_models = {
            AIProvider.GROQ: "llama-3.1-70b-versatile",
            AIProvider.OLLAMA: "llama3.1"
        }
        
        # Initialize the appropriate client
        if provider == AIProvider.GROQ:
            self._init_groq()
        elif provider == AIProvider.OLLAMA:
            self._init_ollama()
        else:
            logger.info("No AI provider configured")
    
    def _init_groq(self):
        """Initialize Groq client"""
        if not self.api_key:
            logger.warning("Groq API key not provided")
            return
        
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            self.is_available = True
            logger.info("✅ Groq API initialized")
        except ImportError:
            logger.error("Groq package not installed. Run: pip install groq")
        except Exception as e:
            logger.error(f"Failed to initialize Groq: {e}")
    
    def _init_ollama(self):
        """Initialize Ollama client"""
        try:
            import ollama
            # Test if Ollama is running
            try:
                ollama.list()
                self.client = ollama
                self.is_available = True
                logger.info("✅ Ollama (Local Llama3) initialized")
            except Exception as e:
                logger.error(f"Ollama not running. Start it with: ollama serve")
                logger.error(f"Error: {e}")
        except ImportError:
            logger.error("Ollama package not installed. Run: pip install ollama")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        **kwargs
    ) -> Optional[str]:
        """
        Unified chat interface - works with both Groq and Ollama.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            AI response text, or None if failed
        """
        if not self.is_available:
            logger.warning("AI provider not available")
            return None
        
        try:
            if self.provider == AIProvider.GROQ:
                return self._chat_groq(messages, temperature, max_tokens, **kwargs)
            elif self.provider == AIProvider.OLLAMA:
                return self._chat_ollama(messages, temperature, max_tokens, **kwargs)
            else:
                return None
        except Exception as e:
            logger.error(f"AI chat failed: {e}")
            return None
    
    def _chat_groq(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Call Groq API"""
        model = self.model or self.default_models[AIProvider.GROQ]
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content.strip()
    
    def _chat_ollama(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Call Ollama (local)"""
        model = self.model or self.default_models[AIProvider.OLLAMA]
        
        response = self.client.chat(
            model=model,
            messages=messages,
            options={
                'temperature': temperature,
                'num_predict': max_tokens,
                **kwargs
            }
        )
        return response['message']['content']
    
    def test_connection(self) -> bool:
        """Test if AI provider is working"""
        if not self.is_available:
            return False
        
        try:
            test_response = self.chat(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say 'OK' if you can read this."}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            if test_response and len(test_response) > 0:
                logger.info(f"✅ AI provider test successful: {self.provider.value}")
                return True
            else:
                logger.warning(f"⚠️ AI provider test failed: empty response")
                return False
                
        except Exception as e:
            logger.error(f"❌ AI provider test failed: {e}")
            return False
    
    def get_provider_name(self) -> str:
        """Get human-readable provider name"""
        names = {
            AIProvider.GROQ: "Groq API (Online)",
            AIProvider.OLLAMA: "Ollama (Local Llama3)",
            AIProvider.NONE: "No AI"
        }
        return names.get(self.provider, "Unknown")
    
    def get_model_name(self) -> str:
        """Get the model being used"""
        if self.provider == AIProvider.NONE:
            return "N/A"
        return self.model or self.default_models.get(self.provider, "Unknown")


def create_ai_client(
    provider_name: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> UnifiedAIClient:
    """
    Factory function to create AI client from string name.
    
    Args:
        provider_name: "groq", "ollama", or "none"
        api_key: API key for Groq (optional)
        model: Model name (optional)
        
    Returns:
        Configured UnifiedAIClient
        
    Example:
        client = create_ai_client("ollama")
        response = client.chat(messages=[...])
    """
    provider_map = {
        "groq": AIProvider.GROQ,
        "ollama": AIProvider.OLLAMA,
        "none": AIProvider.NONE,
        "manual": AIProvider.NONE
    }
    
    provider = provider_map.get(provider_name.lower(), AIProvider.NONE)
    return UnifiedAIClient(provider=provider, api_key=api_key, model=model)


# Compatibility wrapper for existing code
class AIInsightsGeneratorBase:
    """
    Base class that existing AI modules can inherit from.
    Provides unified AI access.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        use_ollama: bool = False,
        model: Optional[str] = None
    ):
        """
        Initialize with either Groq or Ollama.
        
        Args:
            api_key: Groq API key (if using Groq)
            use_ollama: If True, use Ollama instead of Groq
            model: Custom model name (optional)
        """
        if use_ollama:
            provider = AIProvider.OLLAMA
        elif api_key:
            provider = AIProvider.GROQ
        else:
            provider = AIProvider.NONE
        
        self.ai_client = UnifiedAIClient(
            provider=provider,
            api_key=api_key,
            model=model
        )
        
        # Backward compatibility
        self.api_key = api_key
        self.use_ollama = use_ollama
    
    def test_connection(self) -> bool:
        """Test AI connection"""
        return self.ai_client.test_