#!/usr/bin/env python3
"""
Quick Setup Script for Multi-Language Support
Run this to check your setup and configuration
"""

import sys
import os
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_status(check_name, passed, details=""):
    """Print check status"""
    status = "✅" if passed else "❌"
    print(f"{status} {check_name}")
    if details:
        print(f"   {details}")

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    required = (3, 7)
    passed = version >= required
    
    print_status(
        "Python Version",
        passed,
        f"Current: {version.major}.{version.minor}, Required: >={required[0]}.{required[1]}"
    )
    return passed

def check_dependencies():
    """Check required packages"""
    print("\nChecking Dependencies:")
    
    required_packages = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'plotly': 'plotly',
        'yfinance': 'yfinance',
        'groq': 'groq (optional for AI insights)',
    }
    
    optional_packages = {
        'deep_translator': 'deep-translator (optional for auto-translation)',
        'babel': 'babel (optional for number/date formatting)',
    }
    
    all_good = True
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            print_status(description, True)
        except ImportError:
            print_status(description, False, f"Install: pip install {package}")
            all_good = False
    
    print("\nOptional Dependencies:")
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print_status(description, True)
        except ImportError:
            print_status(description, False, f"Optional: pip install {package}")
    
    return all_good

def check_files():
    """Check if required files exist"""
    print("\nChecking Required Files:")
    
    required_files = [
        'translations.py',
        'config.py',
        'ai_insights.py',
    ]
    
    optional_files = [
        'translation_manager.py',
        'translate_helper.py',
        'streamlit_intrinsic_value_v5_multilang.py',
    ]
    
    core_files = [
        'core/Data_fetcher.py',
        'core/IV_simplifier.py',
        'core/ValuationCalculator.py',
        'core/model_selector.py',
    ]
    
    all_good = True
    
    for file in required_files:
        exists = Path(file).exists()
        print_status(file, exists, "" if exists else "Missing! Create this file")
        if not exists:
            all_good = False
    
    print("\nOptional Files:")
    for file in optional_files:
        exists = Path(file).exists()
        print_status(file, exists)
    
    print("\nCore Module Files:")
    for file in core_files:
        exists = Path(file).exists()
        print_status(file, exists)
    
    return all_good

def check_translations():
    """Check translation dictionary"""
    print("\nChecking Translations:")
    
    try:
        from translations import TRANSLATIONS, get_available_languages
        
        languages = get_available_languages()
        print_status(f"Languages available: {len(languages)}", True, f"{list(languages.keys())}")
        
        # Check each language
        en_keys = set(TRANSLATIONS['en'].keys())
        print(f"\n   Total translation keys: {len(en_keys)}")
        
        for lang_code, lang_name in languages.items():
            if lang_code == 'en':
                continue
            
            lang_keys = set(TRANSLATIONS[lang_code].keys())
            missing = len(en_keys - lang_keys)
            completion = (len(lang_keys) / len(en_keys) * 100) if en_keys else 0
            
            status = completion >= 90
            print_status(
                f"{lang_name} ({lang_code})",
                status,
                f"{completion:.1f}% complete ({len(lang_keys)}/{len(en_keys)} keys)"
            )
        
        return True
        
    except ImportError as e:
        print_status("Import translations.py", False, str(e))
        return False
    except Exception as e:
        print_status("Translations validation", False, str(e))
        return False

def check_environment():
    """Check environment variables"""
    print("\nChecking Environment Variables:")
    
    groq_key = os.getenv("GROQ_API_KEY")
    print_status(
        "GROQ_API_KEY",
        groq_key is not None,
        "Set for AI insights" if groq_key else "Optional: Set for AI-powered insights"
    )
    
    default_lang = os.getenv("DEFAULT_LANGUAGE", "en")
    print_status(
        "DEFAULT_LANGUAGE",
        True,
        f"Using: {default_lang}"
    )

def run_quick_test():
    """Run a quick functionality test"""
    print("\nRunning Quick Test:")
    
    try:
        from translations import get_text
        
        # Test basic translation
        text_en = get_text('page_title', 'en')
        text_pt = get_text('page_title', 'pt')
        
        passed = text_en != text_pt and len(text_en) > 0 and len(text_pt) > 0
        print_status(
            "Translation function",
            passed,
            f"EN: '{text_en[:30]}...' | PT: '{text_pt[:30]}...'"
        )
        
        # Test format variables
        text_with_var = get_text('sensitivity_info', 'en', ticker='AAPL', param='rate')
        passed = 'AAPL' in text_with_var and 'rate' in text_with_var
        print_status(
            "Format variables",
            passed,
            f"Result: '{text_with_var[:50]}...'"
        )
        
        return True
        
    except Exception as e:
        print_status("Quick test", False, str(e))
        return False

def print_next_steps(all_checks_passed):
    """Print next steps"""
    print("\n" + "="*70)
    
    if all_checks_passed:
        print("  🎉 ALL CHECKS PASSED!")
        print("="*70)
        print("\n✅ Your multi-language setup is ready!")
        print("\nNext steps:")
        print("  1. Run the main app:")
        print("     streamlit run streamlit_intrinsic_value_v5_multilang.py")
        print("\n  2. Test language switching:")
        print("     - Click the language selector (top-right)")
        print("     - Try each language")
        print("     - Verify all text is translated")
        print("\n  3. Manage translations:")
        print("     streamlit run translation_manager.py")
        print("\n  4. Read the full guide:")
        print("     Open MULTILANG_README.md")
        
    else:
        print("  ⚠️  SETUP INCOMPLETE")
        print("="*70)
        print("\n❌ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  • Install missing packages: pip install -r requirements.txt")
        print("  • Copy missing files from the artifacts")
        print("  • Check file paths and names")
        print("\n📚 See MULTILANG_README.md for detailed instructions")

def main():
    """Main setup check"""
    print_header("🌐 MULTI-LANGUAGE SETUP CHECK")
    
    print("\nThis script will verify your multi-language setup.")
    print("Make sure you've copied all the required files to your project.\n")
    
    checks = []
    
    # Run checks
    checks.append(("Python Version", check_python_version()))
    
    print_header("DEPENDENCIES")
    checks.append(("Dependencies", check_dependencies()))
    
    print_header("FILES")
    checks.append(("Files", check_files()))
    
    print_header("TRANSLATIONS")
    checks.append(("Translations", check_translations()))
    
    print_header("ENVIRONMENT")
    check_environment()
    
    print_header("FUNCTIONALITY")
    checks.append(("Quick Test", run_quick_test()))
    
    # Summary
    all_passed = all(check[1] for check in checks)
    
    print_header("SUMMARY")
    print("\nCheck Results:")
    for name, passed in checks:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name:.<30} {status}")
    
    print_next_steps(all_passed)
    
    print("\n" + "="*70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())