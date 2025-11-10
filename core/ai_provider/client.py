import logging
from typing import List, Dict, Optional, Any
from .enums import AIProvider

logger = logging.getLogger(__name__)

class UnifiedAIClient:
    """Unified interface for AI calls - supports Groq and Ollama."""

    def __init__(self, provider: AIProvider = AIProvider.NONE, api_key: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.client = None
        self.is_available = False

        self.default_models = {
            AIProvider.GROQ: "llama-3.1-70b-versatile",
            AIProvider.OLLAMA: "llama3.1"
        }

        if provider == AIProvider.GROQ:
            self._init_groq()
        elif provider == AIProvider.OLLAMA:
            self._init_ollama()
        else:
            logger.info("No AI provider configured")

    def _init_groq(self):
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
        try:
            import ollama
            ollama.list()  # test if running
            self.client = ollama
            self.is_available = True
            logger.info("✅ Ollama initialized")
        except ImportError:
            logger.error("Ollama package not installed. Run: pip install ollama")
        except Exception as e:
            logger.error(f"Ollama not running. Start with: ollama serve | Error: {e}")

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = 1000, **kwargs) -> Optional[str]:
        if not self.is_available:
            logger.warning("AI provider not available")
            return None
        try:
            if self.provider == AIProvider.GROQ:
                return self._chat_groq(messages, temperature, max_tokens, **kwargs)
            elif self.provider == AIProvider.OLLAMA:
                return self._chat_ollama(messages, temperature, max_tokens, **kwargs)
        except Exception as e:
            logger.error(f"AI chat failed: {e}")
            return None

    def _chat_groq(self, messages, temperature, max_tokens, **kwargs) -> str:
        model = self.model or self.default_models[AIProvider.GROQ]
        response = self.client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
        return response.choices[0].message.content.strip()

    def _chat_ollama(self, messages, temperature, max_tokens, **kwargs) -> str:
        model = self.model or self.default_models[AIProvider.OLLAMA]
        response = self.client.chat(model=model, messages=messages, options={'temperature': temperature, 'num_predict': max_tokens, **kwargs})
        return response['message']['content']

    def test_connection(self) -> bool:
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
            return bool(test_response)
        except Exception as e:
            logger.error(f"AI provider test failed: {e}")
            return False

    def get_provider_name(self) -> str:
        names = {
            AIProvider.GROQ: "Groq API (Online)",
            AIProvider.OLLAMA: "Ollama (Local Llama3)",
            AIProvider.NONE: "No AI"
        }
        return names.get(self.provider, "Unknown")

    def get_model_name(self) -> str:
        return self.model or self.default_models.get(self.provider, "Unknown") if self.provider != AIProvider.NONE else "N/A"
