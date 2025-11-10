import pandas as pd
from typing import Dict, List
from datetime import datetime

def export_to_csv(data: pd.DataFrame, filename: str) -> str:
    """Export DataFrame to CSV string"""
    return data.to_csv(index=False)

def generate_text_report(results: Dict, config: Dict, tickers: List[str]) -> str:
    """Generate comprehensive text report"""
    report_lines = [
        "=" * 80,
        "INTRINSIC VALUE ANALYSIS REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Tickers: {', '.join(tickers)}",
        "",
        "PARAMETERS:",
        f"- Discount Rate: {config['discount_rate']*100:.1f}%",
        f"- Terminal Growth: {config['terminal_growth']*100:.1f}%",
        f"- Target Margin of Safety: {config['margin_of_safety']*100:.1f}%",
        f"- Selection Mode: {'AI Mode' if config.get('use_ai_mode') else 'Auto-select' if config.get('use_smart_selection') else 'Manual'}",
        f"- Competitive Analysis: {'Enabled' if config.get('enable_competitive') else 'Disabled'}",
        f"- ML Predictions: {'Enabled' if config.get('enable_ml_prediction') else 'Disabled'}",
        f"- AI Provider: {config.get('ai_method', 'N/A')}",
        "",
        "=" * 80,
        "INTRINSIC VALUES:",
        "=" * 80,
    ]
    
    # Add valuation table
    if results.get('valuations'):
        df_iv = pd.DataFrame(results['valuations'])
        report_lines.append(df_iv.to_string())
        report_lines.append("")
    
    # Add margin of safety
    if results.get('margins'):
        report_lines.extend([
            "=" * 80,
            "MARGIN OF SAFETY ANALYSIS:",
            "=" * 80,
        ])
        df_margins = pd.DataFrame(results['margins'])
        report_lines.append(df_margins.to_string())
        report_lines.append("")
    
    # Add competitive comparison
    if results.get('competitive'):
        report_lines.extend([
            "=" * 80,
            "COMPETITIVE COMPARISON:",
            "=" * 80,
        ])
        for ticker, comp_data in results['competitive'].items():
            report_lines.append(f"\n{ticker} COMPETITORS: {', '.join(comp_data['competitors'])}")
            report_lines.append(comp_data['table'].to_string())
            report_lines.append("")
    
    # Add ML predictions
    if results.get('ml_predictions'):
        report_lines.extend([
            "=" * 80,
            "ML PRICE PREDICTIONS:",
            "=" * 80,
        ])
        for ticker, pred in results['ml_predictions'].items():
            report_lines.extend([
                f"\n{ticker}:",
                f"  Direction: {pred['prediction']}",
                f"  Expected Change: {pred['expected_change']}",
                f"  Confidence: {pred['confidence']}",
                f"  Model: {pred['model_used']}",
                f"  Sector: {pred['sector']}",
                ""
            ])
    
    # Add AI insights
    if results.get('ai_insights'):
        report_lines.extend([
            "=" * 80,
            "AI INSIGHTS:",
            "=" * 80,
        ])
        for ticker, insight in results['ai_insights'].items():
            report_lines.append(f"\n{ticker}:")
            report_lines.append(insight)
            report_lines.append("")
    
    # Add warnings
    if results.get('warnings'):
        report_lines.extend([
            "=" * 80,
            "DATA QUALITY WARNINGS:",
            "=" * 80,
        ])
        for ticker, warnings in results['warnings'].items():
            if warnings:
                report_lines.append(f"\n{ticker}:")
                for warning in warnings:
                    report_lines.append(f"  - {warning}")
                report_lines.append("")
    
    # Footer
    report_lines.extend([
        "=" * 80,
        "DISCLAIMER:",
        "=" * 80,
        "This analysis is for informational purposes only and should not be",
        "considered as financial advice. Always conduct your own research and",
        "consult with a qualified financial advisor before making investment decisions.",
        "=" * 80
    ])
    
    return "\n".join(report_lines)
