import streamlit as st
from translations.translator import get_text

def render_ai_config():
    """Render AI configuration section"""
    st.sidebar.markdown("---")
    st.sidebar.subheader(get_text('ai_insights'))
    
    enable_ai = st.sidebar.checkbox(
        get_text('enable_ai'),
        value=False,
        help="Enable AI-powered insights and analysis"
    )
    
    config = {
        'enable_ai_insights': enable_ai,
        'use_ollama': False,
        'groq_api_key': None,
        'ai_method': "Manual Only"
    }
    
    if enable_ai:
        ai_method = st.sidebar.radio(
            "AI Provider",
            ["Groq API (Online)", "Ollama (Local - Free)", "Manual Only"],
            help="Choose how to power AI features"
        )
        config['ai_method'] = ai_method
        
        if ai_method == "Ollama (Local - Free)":
            config['use_ollama'] = True
            st.sidebar.info("💻 Using local Llama3 via Ollama")
            st.sidebar.caption("""
**Setup:**
1. Install: `ollama.ai/download`
2. Pull model: `ollama pull llama3.1`
3. Start: `ollama serve` (in terminal)
""")
        elif ai_method == "Groq API (Online)":
            api_key = st.sidebar.text_input(
                get_text('groq_key'),
                type="password",
                placeholder="gsk_...",
                help="Get free API key from console.groq.com"
            )
            if api_key:
                config['groq_api_key'] = api_key
                st.sidebar.success("✅ API key provided")
            else:
                st.sidebar.warning("⚠️ Groq API key required")
        else:
            config['enable_ai_insights'] = False
            st.sidebar.info("📋 Using manual competitor mappings")
    
    return config

