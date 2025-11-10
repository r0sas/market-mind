import streamlit as st
from translations.translator import get_text

def render_information_sections():
    """Render information expanders at bottom of page"""
    st.markdown("---")
    
    with st.expander(get_text('about_models')):
        render_models_info()
    
    with st.expander(get_text('faq')):
        render_faq()
    
    with st.expander(get_text('technical')):
        render_technical_details()
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<p style='text-align: center; color: gray;'>{get_text('disclaimer')}</p>",
        unsafe_allow_html=True
    )

def render_models_info():
    """Render valuation models information"""
    st.markdown("""
    ### Valuation Models Explained
    
    **1. Discounted Cash Flow (DCF)**
    - Projects future free cash flows and discounts them to present value
    - Best for: Companies with stable, positive cash flows
    - **Auto-Selected When:** Company has positive FCF in 60%+ of years
    
    **2. Dividend Discount Model (DDM)**
    - Values stock based on present value of future dividends
    - **Single-Stage:** For stable dividend payers (growth <8%)
    - **Multi-Stage:** For growing dividend payers (growth ≥8%)
    - **Auto-Selected When:** Company pays dividends consistently
    
    **3. P/E Multiplier Model**
    - Uses average historical P/E ratio × current EPS
    - Best for: Stable, profitable companies
    - **Auto-Selected When:** Positive earnings in 60%+ of years
    
    **4. Asset-Based Valuation**
    - Book value (Total Assets - Total Liabilities) per share
    - **Auto-Selected When:** Low profitability but strong asset base
    
    **5. Modern Graham Formula**
    - Benjamin Graham's formula adjusted for bond yields
    - **Auto-Selected When:** 5+ years of profitability, strong ROE, low debt
    
    **6. ML Price Prediction**
    - Uses sector-specific machine learning models trained on S&P 500 data
    - Predicts 12-month price direction based on 15-20 financial metrics
    - **Features:** Sector-specific models, confidence scoring, combined analysis with intrinsic value
    """)

def render_faq():
    """Render FAQ section"""
    st.markdown("""
    **Q: What's the difference between the three modes?**  
    A: 
    - **Manual**: You choose models and set parameters
    - **Auto-select**: AI chooses models, you set parameters
    - **AI Mode**: AI chooses models AND optimizes parameters per company
    
    **Q: Why do different models give different values?**  
    A: Each model uses different assumptions. Look at the range and average.
    
    **Q: What is margin of safety?**  
    A: It's like a "safety cushion" - how much cheaper the stock is than its estimated value.
    
    **Q: How does competitive comparison work?**  
    A: AI detects the top 2 competitors and shows side-by-side performance metrics, price charts, and financial ratios.
    
    **Q: Can I specify my own competitors?**  
    A: Yes! Use the "Manual Competitors" field in the sidebar to override AI detection.
    
    **Q: What's the difference between Groq and Ollama?**  
    A:
    - **Groq API**: Fast online AI, requires free API key, generous free tier
    - **Ollama**: Free local AI, runs on your computer, no API needed, fully private
    - **Manual Only**: Uses pre-programmed competitor mappings, no AI required
    
    **Q: How accurate are the ML predictions?**  
    A: ML models are trained on 10+ years of S&P 500 data with sector-specific tuning. Accuracy varies by sector and market conditions. Always combine with fundamental analysis.
    
    **Q: How often should I recalculate?**  
    A: Monthly or quarterly is sufficient for long-term investing. Recalculate after major news or earnings releases.
    """)

def render_technical_details():
    """Render technical details"""
    from config.display_constants import DEFAULT_DISCOUNT_RATE, DEFAULT_TERMINAL_GROWTH, DEFAULT_MARGIN_OF_SAFETY
    
    st.markdown(f"""
    ### Default Parameters
    - **Discount Rate**: {DEFAULT_DISCOUNT_RATE*100:.1f}%
    - **Terminal Growth Rate**: {DEFAULT_TERMINAL_GROWTH*100:.1f}%
    - **Margin of Safety**: {DEFAULT_MARGIN_OF_SAFETY*100:.0f}%
    - **Data Source**: Yahoo Finance via yfinance library
    
    ### AI Mode Features
    - Company-specific discount rates (6-20%)
    - Industry-adjusted parameters
    - Risk-based optimization
    - Beginner-friendly explanations
    
    ### ML Prediction Features
    - Sector-specific trained models
    - 15-20 financial metrics per prediction
    - Confidence scoring (High/Medium/Low)
    - Combined analysis with intrinsic value
    
    ### Competitive Comparison
    - AI-powered competitor detection using Groq/Ollama LLM
    - Fallback to sector-based detection if AI unavailable
    - Price performance (3M, 6M, 1Y)
    - Key metrics radar chart
    - Detailed financial statistics
    
    ### Ollama Setup (Local AI)
    ```bash
    # 1. Install Ollama
    # Download from: https://ollama.ai/download
    
    # 2. Pull Llama3.1 model
    ollama pull llama3.1
    
    # 3. Start Ollama server
    ollama serve
    
    # 4. Install Python package
    pip install ollama
    
    # 5. Select "Ollama (Local - Free)" in sidebar
    ```
    
    ### Model Selection Algorithm
    - Analyzes 10+ data quality metrics per model
    - Checks data completeness, consistency, and trends
    - Assigns fit scores (0.0 - 1.0) based on suitability
    - Recommends models above minimum threshold
    
    ### Data Requirements
    - **Minimum**: 5 years of historical data
    - **Recommended**: 10+ years for trend analysis
    - **Required Fields**: Revenue, Net Income, Cash Flow, Assets, Liabilities
    - **Optional Fields**: Dividends, EPS, Share Price
    
    ### Calculation Methods
    - **DCF**: 5-year projection + terminal value
    - **DDM**: Gordon Growth Model variants
    - **P/E Multiple**: 5-year average P/E ratio
    - **Asset-Based**: Book value per share
    - **Graham**: EPS × (8.5 + 2g) × 4.4/Y
    
    ### API Rate Limits
    - **Yahoo Finance**: ~2000 requests/hour
    - **Groq API**: 30 requests/minute (free tier)
    - **Ollama**: No limits (local)
    
    ### Caching
    - Stock data cached for 1 hour
    - ML models loaded once per session
    - Competitor data cached for session
    """)
