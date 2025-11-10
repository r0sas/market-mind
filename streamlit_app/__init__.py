"""
Streamlit Intrinsic Value Calculator Application

Modular, maintainable dashboard for stock valuation analysis.
"""

__version__ = "5.0.0"
__author__ = "JT"

if __name__ == "__main__":
    from .main import main
    main()
    
__all__ = ["main"]