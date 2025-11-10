import streamlit as st

def render_competitive_config():
    """Render competitive comparison configuration"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏆 Competitive Analysis")
    
    enable = st.sidebar.checkbox(
        "Enable Competitor Comparison",
        value=False,
        help="Compare with AI-detected competitors"
    )
    
    config = {'enable_competitive': enable, 'manual_competitors': None}
    
    if enable:
        st.sidebar.info("💡 Works best for single-stock analysis")
        
        manual_input = st.sidebar.text_input(
            "Manual Competitors (Optional)",
            placeholder="MSFT, GOOGL",
            help="Leave empty for AI detection, or enter 1-2 tickers"
        )
        
        if manual_input and manual_input.strip():
            config['manual_competitors'] = [
                c.strip().upper() for c in manual_input.split(',') if c.strip()
            ]
    
    return config

