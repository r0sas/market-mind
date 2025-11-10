import streamlit as st

def render_ml_config():
    """Render ML prediction configuration"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 ML Price Prediction")
    
    enable = st.sidebar.checkbox(
        "Enable ML Price Prediction",
        value=False,
        help="Use machine learning to predict if stock price will go up or down"
    )
    
    if enable:
        st.sidebar.info("💡 Uses sector-specific trained models")
        st.sidebar.caption("Predictions based on historical S&P 500 patterns")
    
    return {'enable_ml_prediction': enable}

