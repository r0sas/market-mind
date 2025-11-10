import streamlit as st
from typing import Dict

def display_competitive_comparison(competitive_results: Dict):
    """Display competitive comparison section"""
    st.markdown("### 🏆 Competitive Comparison")
    st.markdown("*AI-powered competitor analysis*")
    
    for ticker, comp_data in competitive_results.items():
        st.markdown(f"#### {ticker} vs Competitors")
        
        # Display summary
        with st.expander(f"📝 {ticker} Competitive Summary", expanded=True):
            st.markdown(comp_data['summary'])
            st.caption(f"**Detected Competitors:** {', '.join(comp_data['competitors'])}")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Comparison Table",
            "📈 Price Performance",
            "🎯 Metrics Radar",
            "💹 Detailed Stats"
        ])
        
        with tab1:
            display_comparison_table(comp_data)
        
        with tab2:
            display_price_performance(comp_data)
        
        with tab3:
            display_metrics_radar(comp_data)
        
        with tab4:
            display_detailed_stats(comp_data, ticker)
        
        st.markdown("---")

def display_comparison_table(comp_data: Dict):
    """Display side-by-side comparison table"""
    st.markdown("##### Side-by-Side Comparison")
    
    display_table = comp_data['table'].copy()
    
    # Try to apply gradient styling
    try:
        display_table['3M_numeric'] = display_table['3M Return'].str.rstrip('%').astype(float)
        display_table['6M_numeric'] = display_table['6M Return'].str.rstrip('%').astype(float)
        display_table['1Y_numeric'] = display_table['1Y Return'].str.rstrip('%').astype(float)
        
        styled_table = display_table.style.background_gradient(
            subset=['3M_numeric', '6M_numeric', '1Y_numeric'],
            cmap='RdYlGn',
            vmin=-30,
            vmax=30
        )
        
        styled_table = styled_table.hide(axis='columns', subset=['3M_numeric', '6M_numeric', '1Y_numeric'])
        
    except Exception:
        styled_table = display_table.style
    
    st.dataframe(styled_table, use_container_width=True, hide_index=True)
    
    # Highlight best/worst
    try:
        returns_1y = comp_data['table']['1Y Return'].str.rstrip('%').astype(float)
        best_idx = returns_1y.idxmax()
        worst_idx = returns_1y.idxmin()
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"🏆 **Best 1Y:** {comp_data['table'].loc[best_idx, 'Ticker']} ({comp_data['table'].loc[best_idx, '1Y Return']})")
        with col2:
            st.error(f"📉 **Weakest 1Y:** {comp_data['table'].loc[worst_idx, 'Ticker']} ({comp_data['table'].loc[worst_idx, '1Y Return']})")
    except Exception:
        pass

def display_price_performance(comp_data: Dict):
    """Display price performance charts"""
    st.markdown("##### Price Performance Over Time")
    
    period_tab1, period_tab2, period_tab3 = st.tabs(["1 Year", "6 Months", "3 Months"])
    
    with period_tab1:
        st.plotly_chart(comp_data['chart_1y'], use_container_width=True)
    
    with period_tab2:
        st.plotly_chart(comp_data['chart_6m'], use_container_width=True)
    
    with period_tab3:
        st.plotly_chart(comp_data['chart_3m'], use_container_width=True)

def display_metrics_radar(comp_data: Dict):
    """Display metrics radar chart"""
    st.markdown("##### Key Metrics Comparison")
    st.plotly_chart(comp_data['radar'], use_container_width=True)
    st.info("💡 **How to read:** Larger areas indicate stronger performance. The main stock is highlighted.")

def display_detailed_stats(comp_data: Dict, ticker: str):
    """Display detailed financial statistics"""
    st.markdown("##### Detailed Financial Metrics")
    
    detailed_data = []
    for ticker_key in [ticker] + comp_data['competitors']:
        if ticker_key in comp_data['data']:
            d = comp_data['data'][ticker_key]
            detailed_data.append({
                'Ticker': ticker_key,
                'Company': d['company_name'],
                'Sector': d['sector'],
                'Market Cap': f"${d['market_cap']/1e9:.2f}B" if d['market_cap'] else 'N/A',
                'P/E': f"{d['pe_ratio']:.2f}" if d['pe_ratio'] else 'N/A',
                'Forward P/E': f"{d['forward_pe']:.2f}" if d['forward_pe'] else 'N/A',
                'ROE': f"{d['roe']*100:.1f}%" if d['roe'] else 'N/A',
                'Profit Margin': f"{d['profit_margin']*100:.1f}%" if d['profit_margin'] else 'N/A',
                'Debt/Equity': f"{d['debt_to_equity']:.2f}" if d['debt_to_equity'] else 'N/A',
                'Beta': f"{d['beta']:.2f}" if d['beta'] else 'N/A'
            })
    
    if detailed_data:
        import pandas as pd
        df_detailed = pd.DataFrame(detailed_data)
        st.dataframe(df_detailed, use_container_width=True, hide_index=True)
