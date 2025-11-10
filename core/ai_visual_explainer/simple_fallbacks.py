from typing import Dict, List

def generate_simple_table_explanation(ticker: str, current_price: float, valuations: Dict[str, float], average: float) -> str:
    diff = average - current_price
    diff_pct = (diff / current_price) * 100
    if diff_pct > 20:
        verdict = f"appears undervalued - trading at ${current_price:.2f} but worth around ${average:.2f}"
    elif diff_pct < -20:
        verdict = f"appears overvalued - trading at ${current_price:.2f} but worth around ${average:.2f}"
    else:
        verdict = f"is fairly valued - trading at ${current_price:.2f}, close to its estimated worth of ${average:.2f}"
    spread = max(valuations.values()) - min(valuations.values())
    agreement = "There's significant disagreement between models." if spread/average > 0.3 else "Models mostly agree on the value."
    return f"**What This Means:** {ticker} {verdict}. {agreement}"

def generate_simple_comparison_explanation(tickers: List[str], valuations: Dict[str, Dict], current_prices: Dict[str, float]) -> str:
    undervalued, overvalued = [], []
    for t in tickers:
        avg = sum(valuations[t].values()) / len(valuations[t])
        current = current_prices[t]
        if avg > current * 1.15:
            undervalued.append(t)
        elif current > avg * 1.15:
            overvalued.append(t)
    text = f"**Comparing {len(tickers)} Stocks:** "
    if undervalued:
        text += f"{', '.join(undervalued)} look like bargains. "
    if overvalued:
        text += f"{', '.join(overvalued)} seem expensive. "
    text += "Remember: intrinsic value is just one factor."
    return text

def generate_simple_margin_explanation(ticker: str, margin_pct: float, target: float, is_good: bool) -> str:
    analogy = f"Think of it like buying a $100 item on sale with a {target*100:.0f}% discount."
    verdict = f"{ticker} has a {margin_pct:.0f}% margin of safety." if is_good else f"{ticker}'s margin is below target."
    return f"**Margin of Safety Explained:** {analogy} {verdict}"

def generate_simple_selection_explanation(ticker: str, selected: List[str], excluded: List[str]) -> str:
    return f"**Why Different Models?** For {ticker}, {len(selected)} models are used while {len(excluded)} don't fit. Using the right tools gives more accurate results!"

def generate_simple_parameter_explanation(ticker: str, discount_rate: float, terminal_growth: float) -> str:
    if discount_rate < 0.09:
        risk = "low-risk"
    elif discount_rate < 0.12:
        risk = "moderate-risk"
    else:
        risk = "higher-risk"
    return f"**Risk Settings:** {ticker} has discount rate {discount_rate:.1%} ({risk}). Terminal growth is {terminal_growth:.1%}."
