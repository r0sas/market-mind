"""
Translation Helper Script
Auto-translates English keys to other languages using Groq API or deep-translator
"""

import json
import os
from typing import Dict
from groq import Groq

# Option 1: Using Groq API (if you have it)
def translate_with_groq(text: str, target_lang: str, api_key: str) -> str:
    """Translate text using Groq API"""
    client = Groq(api_key=api_key)
    
    lang_map = {
        'pt': 'Portuguese',
        'es': 'Spanish',
        'fr': 'French',
        'it': 'Italian',
        'de': 'German'
    }
    
    target_language = lang_map.get(target_lang, target_lang)
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a professional translator. Translate the following text to {target_language}. Preserve any formatting like ** for bold, emojis, and technical terms. Only return the translation, nothing else."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            max_tokens=200,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error translating: {e}")
        return text


# Option 2: Using deep-translator (no API key needed)
def translate_with_deep_translator(text: str, target_lang: str) -> str:
    """Translate text using deep-translator (Google Translate)"""
    try:
        from deep_translator import GoogleTranslator
        
        translator = GoogleTranslator(source='en', target=target_lang)
        return translator.translate(text)
    except ImportError:
        print("deep-translator not installed. Run: pip install deep-translator")
        return text
    except Exception as e:
        print(f"Translation error: {e}")
        return text


def translate_dict(english_dict: Dict, target_lang: str, method: str = 'groq', api_key: str = None) -> Dict:
    """
    Translate entire dictionary
    
    Args:
        english_dict: Dictionary with English text
        target_lang: Target language code (pt, es, fr, it, de)
        method: 'groq' or 'deep_translator'
        api_key: Groq API key (only needed for 'groq' method)
    """
    translated = {}
    total = len(english_dict)
    
    for idx, (key, value) in enumerate(english_dict.items(), 1):
        print(f"[{idx}/{total}] Translating '{key}'...")
        
        if method == 'groq' and api_key:
            translated[key] = translate_with_groq(value, target_lang, api_key)
        else:
            translated[key] = translate_with_deep_translator(value, target_lang)
    
    return translated


def complete_french_translation(api_key: str = None):
    """Complete French translation"""
    
    english_base = {
        'page_subtitle': 'Calculate intrinsic value using multiple valuation models: DCF, DDM, P/E Model, Asset-Based, and Graham Formula',
        'model_selection_question': 'How should models be selected?',
        'auto_select': '🤖 Auto-select (Recommended)',
        'manual_select': '✋ Manual selection',
        'auto_select_help': 'Auto-select uses AI to choose the best models for each company',
        'smart_enabled': '✨ Smart selection enabled',
        'manual_active': 'Manual selection active',
        'min_fit_score': 'Minimum Model Fit Score',
        'min_fit_score_help': 'Only show models scoring above this threshold (0.5 = recommended)',
        'show_excluded': 'Show excluded models',
        'show_excluded_help': 'Display fit scores for models that didn\'t meet the threshold',
        'model_selection': 'Model Selection',
        'select_models': 'Select Valuation Models',
        'select_models_help': 'Choose which valuation models to calculate and display',
        'advanced_params': 'Advanced Parameters',
        'dcf_params': '⚙️ DCF Parameters',
        'discount_rate': 'Discount Rate (%)',
        'discount_rate_help': 'Required rate of return (WACC)',
        'terminal_growth': 'Terminal Growth Rate (%)',
        'terminal_growth_help': 'Perpetual growth rate for terminal value',
        'analysis_options': '📈 Analysis Options',
        'show_confidence': 'Show Confidence Scores',
        'show_warnings': 'Show Model Warnings',
        'weighted_avg': 'Use Weighted Average',
        'weighted_avg_help': 'Weight models by confidence score',
        'margin_safety': 'Target Margin of Safety (%)',
        'margin_safety_help': 'Desired discount from intrinsic value',
        # Add more as needed...
    }
    
    if api_key:
        return translate_dict(english_base, 'fr', method='groq', api_key=api_key)
    else:
        return translate_dict(english_base, 'fr', method='deep_translator')


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("TRANSLATION HELPER")
    print("="*70)
    
    # Get API key from environment or user input
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("\nNo GROQ_API_KEY found in environment.")
        print("\nOptions:")
        print("1. Enter Groq API key (for better translations)")
        print("2. Use Google Translate (free, no API key needed)")
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == '1':
            api_key = input("Enter Groq API key: ").strip()
            method = 'groq'
        else:
            method = 'deep_translator'
            print("\nUsing Google Translate (install: pip install deep-translator)")
    else:
        method = 'groq'
        print(f"\n✅ Using Groq API")
    
    # Example: Translate a few keys to French
    sample_text = {
        'page_title': 'Intrinsic Value Calculator',
        'calculate': '🔍 Calculate',
        'success': '✅ Successfully analyzed',
    }
    
    print("\nTranslating sample text to French...")
    
    if method == 'groq' and api_key:
        french = translate_dict(sample_text, 'fr', method='groq', api_key=api_key)
    else:
        french = translate_dict(sample_text, 'fr', method='deep_translator')
    
    print("\n" + "="*70)
    print("FRENCH TRANSLATIONS")
    print("="*70)
    for key, value in french.items():
        print(f"{key}: {value}")
    
    print("\n💡 Copy these translations to your translations.py file!")
    print("\n📝 To translate all languages, run:")
    print("   for lang in ['fr', 'it', 'de']:")
    print("       translations = translate_dict(english_dict, lang, 'groq', api_key)")