import streamlit as st
import pandas as pd
from typing import Dict, List
from visualizations.metrics import display_combined_signal

def display_ml_predictions(predictions: Dict, valuations: List[Dict]):
    """Display ML predictions section"""
    st.markdown("### 🤖 ML-Based Price Predictions")
    st.markdown("*Machine learning predictions based on historical S&P 500 patterns*")
    
    # Create prediction summary table
    pred_data = []
    for ticker, pred in predictions.items():
        pred_data.append({
            'Ticker': ticker,
            'Current Price': pred.get('current_price', 'N/A'),
            'ML Prediction': pred['prediction'],
            'Expected Change': pred['expected_change'],
            'Confidence': pred['confidence'],
            'Model Used': pred['model_used'],
            'Sector': pred['sector']
        })
    
    if pred_data:
        df_predictions = pd.DataFrame(pred_data)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(
                df_predictions.style.applymap(
                    lambda x: 'background-color: #d4edda' if 'UP' in str(x) else 
                             'background-color: #f8d7da' if 'DOWN' in str(x) else '',
                    subset=['ML Prediction']
                ),
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            display_prediction_summary(pred_data)
        
        # Explanation
        with st.expander("ℹ️ How ML Predictions Work"):
            st.markdown("""
            **ML Prediction System:**
            
            1. **Training Data**: Historical S&P 500 company data (10+ years)
            2. **Features Used**: 15-20 key financial metrics per sector
            3. **Model Type**: Sector-specific Random Forest or Gradient Boosting
            4. **Prediction**: Expected price change in next 12 months
            
            **Interpretation:**
            - **UP ⬆️**: Model predicts >5% price increase
            - **DOWN ⬇️**: Model predicts >5% price decrease  
            - **FLAT ➡️**: Model predicts -5% to +5% change
            
            **Confidence Levels:**
            - **High**: Model predictions agree (low variance)
            - **Medium**: Some model disagreement
            - **Low**: High uncertainty (model predictions vary)
            
            ⚠️ **Disclaimer**: ML predictions are based on historical patterns and should not be the sole basis for investment decisions.
            """)
        
        # Combined Analysis (ML + IV)
        display_combined_analysis(predictions, valuations)

def display_prediction_summary(pred_data: List[Dict]):
    """Display summary statistics for predictions"""
    up_count = sum(1 for p in pred_data if 'UP' in p['ML Prediction'])
    down_count = sum(1 for p in pred_data if 'DOWN' in p['ML Prediction'])
    
    st.metric("Bullish Predictions", up_count)
    st.metric("Bearish Predictions", down_count)
    
    if up_count > down_count:
        st.success("📈 Overall: **BULLISH**")
    elif down_count > up_count:
        st.error("📉 Overall: **BEARISH**")
    else:
        st.info("➡️ Overall: **NEUTRAL**")

def display_combined_analysis(predictions: Dict, valuations: List[Dict]):
    """Display combined ML + IV analysis"""
    st.markdown("#### 🔍 ML Prediction vs Intrinsic Value Analysis")
    
    # Create valuation lookup
    val_lookup = {v['ticker']: v for v in valuations}
    
    for ticker, ml_pred in predictions.items():
        if ticker in val_lookup:
            valuation = val_lookup[ticker]
            current = valuation.get('current_price', 0)
            avg_iv = valuation.get('average_iv', current)
            
            # Calculate IV signal
            iv_diff_pct = ((avg_iv - current) / current) * 100 if current > 0 else 0
            
            with st.expander(f"🎯 {ticker} - Combined Analysis"):
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.markdown("**📊 Intrinsic Value**")
                    st.metric("Current Price", f"${current:.2f}")
                    st.metric("Average IV", f"${avg_iv:.2f}", delta=f"{iv_diff_pct:+.1f}%")
                
                with col_b:
                    st.markdown("**🤖 ML Prediction**")
                    st.metric("Direction", ml_pred['prediction'])
                    st.metric("Expected Change", ml_pred['expected_change'])
                    st.caption(f"Confidence: {ml_pred['confidence']}")
                
                with col_c:
                    st.markdown("**💡 Combined Signal**")
                    display_combined_signal(iv_diff_pct, ml_pred)

