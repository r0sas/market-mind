import streamlit as st
from translations.en import TRANSLATIONS_EN
from translations.es import TRANSLATIONS_ES

# Language registry
LANGUAGES = {
    'en': TRANSLATIONS_EN,
    'es': TRANSLATIONS_ES
}

def get_text(key: str, lang: str = None, **kwargs) -> str:
    """
    Get translated text for given key.
    
    Args:
        key: Translation key
        lang: Language code (defaults to session state)
        **kwargs: Format parameters for string interpolation
        
    Returns:
        Translated text
    """
    if lang is None:
        lang = st.session_state.get('language', 'en')
    
    # Get translation dict for language
    translations = LANGUAGES.get(lang, TRANSLATIONS_EN)
    
    # Get translated text
    text = translations.get(key, key)
    
    # Apply formatting if kwargs provided
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass  # Return unformatted if error
    
    return text

def get_available_languages() -> dict:
    """Return available languages"""
    return {
        'en': 'English',
        'es': 'Español'
    }

