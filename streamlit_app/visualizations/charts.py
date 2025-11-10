import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict
from config.display_constants import CHART_COLORS

def create_iv_comparison_chart(valuations: List[Dict]) -> go.Figure:
    """Create intrinsic value comparison bar chart"""
    df = pd.DataFrame(valuations)
    
    # Get value columns (exclude metadata)
    exclude_cols = ['Ticker', 'ticker', 'current_price', 'average_iv', 'ml_prediction']
    value_cols = [c for c in df.columns if c not in exclude_cols and not c.endswith('_Confidence')]
    
    # Melt for plotting
    df_plot = df[['Ticker'] + value_cols].melt(
        id_vars='Ticker',
        var_name='Model',
        value_name='Value'
    )
    
    fig = px.bar(
        df_plot,
        x='Ticker',
        y='Value',
        color='Model',
        barmode='group',
        title='Intrinsic Value Comparison',
        labels={'Value': 'Value ($)', 'Model': 'Valuation Model'}
    )
    
    # Add current price lines
    for _, row in df.iterrows():
        if 'current_price' in row and row['current_price']:
            fig.add_hline(
                y=row['current_price'],
                line_dash='dash',
                line_color='red',
                annotation_text=f"{row['Ticker']} Current: ${row['current_price']:.2f}",
                annotation_position='right'
            )
    
    fig.update_layout(height=500)
    return fig

def create_sensitivity_plot(sensitivity_data: Dict, model_name: str) -> go.Figure:
    """Create sensitivity analysis plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sensitivity_data['values'],
        y=sensitivity_data['valuations'],
        mode='lines+markers',
        name=model_name,
        line=dict(color=CHART_COLORS['primary'], width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=f"Sensitivity Analysis: {model_name}",
        xaxis_title=sensitivity_data['parameter'].replace('_', ' ').title(),
        yaxis_title="Intrinsic Value ($)",
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_margin_chart(margin_data: List[Dict], target_margin: float) -> go.Figure:
    """Create margin of safety bar chart"""
    df = pd.DataFrame(margin_data)
    
    df["Color"] = df["Status"].apply(
        lambda x: "Undervalued" if "✓" in x else "Overvalued"
    )
    
    fig = px.bar(
        df,
        x="Ticker",
        y="Margin of Safety (%)",
        color="Color",
        text="Margin of Safety (%)",
        facet_col="Model",
        facet_col_wrap=3,
        color_discrete_map={
            "Undervalued": CHART_COLORS['success'],
            "Overvalued": CHART_COLORS['danger']
        },
        title="Margin of Safety Analysis",
        labels={"Margin of Safety (%)": "Margin of Safety (%)"}
    )
    
    fig.add_hline(
        y=target_margin * 100,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Target: {target_margin*100:.0f}%"
    )
    
    fig.update_traces(textposition='outside', texttemplate='%{text:.1f}%')
    fig.update_layout(height=400, showlegend=True)
    fig.for_each_xaxis(lambda axis: axis.update(tickangle=45))
    
    return fig

def create_fit_score_chart(fit_scores: Dict, min_score: float, ticker: str) -> go.Figure:
    """Create model fit scores bar chart"""
    from core.config import MODEL_DISPLAY_NAMES
    
    fit_data = []
    for model, score in fit_scores.items():
        model_display = MODEL_DISPLAY_NAMES.get(model, model)
        fit_data.append({
            'Model': model_display,
            'Score': score
        })
    
    df = pd.DataFrame(fit_data).sort_values('Score', ascending=False)
    
    # Color based on score
    colors = []
    for score in df['Score']:
        if score >= 0.7:
            colors.append(CHART_COLORS['success'])
        elif score >= 0.5:
            colors.append(CHART_COLORS['warning'])
        elif score >= 0.3:
            colors.append('#fd7e14')  # Orange
        else:
            colors.append(CHART_COLORS['danger'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df['Model'],
        x=df['Score'],
        orientation='h',
        marker=dict(color=colors),
        text=df['Score'].apply(lambda x: f'{x:.2f}'),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Fit Score: %{x:.2f}<extra></extra>'
    ))
    
    fig.add_vline(
        x=min_score,
        line_dash="dash",
        line_color="orange",
        line_width=2,
        annotation_text=f"Target ({min_score:.1f})",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=f"{ticker} - Model Fit Scores",
        xaxis_title="Fit Score (0.0 - 1.0)",
        yaxis_title="",
        height=350,
        showlegend=False,
        xaxis=dict(range=[0, 1.1])
    )
    
    return fig
