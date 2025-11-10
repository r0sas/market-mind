from enum import Enum

class AIProvider(Enum):
    """Available AI providers"""
    GROQ = "groq"
    OLLAMA = "ollama"
    NONE = "none"
