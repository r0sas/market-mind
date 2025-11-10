def build_ai_prompt(ticker, company_data, financial_metrics, rule_based):
    revenue_cagr = financial_metrics.get('revenue_cagr', 'N/A')
    fcf_cagr = financial_metrics.get('fcf_cagr', 'N/A')
    roe = financial_metrics.get('roe', 'N/A')
    net_margin = financial_metrics.get('net_margin', 'N/A')

    prompt = f"""Analyze {ticker} and determine optimal DCF parameters:
Sector: {company_data.get('sector','Unknown')}
Industry: {company_data.get('industry','Unknown')}
Market Cap: ${company_data.get('marketCap',0):,.0f}

Metrics:
- Beta: {financial_metrics.get('beta','N/A')}
- Debt/Equity: {financial_metrics.get('debt_to_equity','N/A')}
- Revenue Growth: {revenue_cagr if isinstance(revenue_cagr,(int,float)) else 'N/A'}
- FCF Growth: {fcf_cagr if isinstance(fcf_cagr,(int,float)) else 'N/A'}
- ROE: {roe if isinstance(roe,(int,float)) else 'N/A'}
- Net Margin: {net_margin if isinstance(net_margin,(int,float)) else 'N/A'}

Rule-based baseline:
- Discount Rate: {rule_based['discount_rate']:.2%}
- Terminal Growth: {rule_based['terminal_growth']:.2%}
- Projection Years: {rule_based['projection_years']}

Respond with JSON only:
{{
  "discount_rate": 0.10,
  "terminal_growth": 0.025,
  "projection_years": 5,
  "explanation": "Brief explanation",
  "reasoning": "Detailed reasoning"
}}"""
    return prompt

def build_rule_based_explanation(ticker, sector, discount_rate, terminal_growth,
                                 base_rate, size_adj, beta_adj, leverage_adj,
                                 market_cap, beta, debt_to_equity, revenue_growth,
                                 market_phase):
    size_category = "large-cap" if market_cap > 10e9 else "mid-cap" if market_cap > 2e9 else "small-cap"
    explanation = f"""**{ticker} Analysis:**
Discount Rate: {discount_rate:.2%} (Base {base_rate:.2%}, Size {size_adj:.2%}, Beta {beta_adj:.2%}, Leverage {leverage_adj:.2%})
Terminal Growth: {terminal_growth:.2%} ({market_phase})
"""
    if size_adj>0: explanation += f"- Higher risk for {size_category}\n"
    if beta>1.2: explanation += "- High beta\n"
    if beta<0.8: explanation += "- Low beta\n"
    if leverage_adj>0: explanation += "- High leverage\n"
    if revenue_growth>0.15: explanation += "- Strong growth supports longer projection\n"
    return explanation
