import logging
logger = logging.getLogger(__name__)

def calculate_wacc(company_data: dict, financial_metrics: dict) -> float:
    try:
        market_cap = company_data.get('marketCap', 0)
        total_debt = financial_metrics.get('total_debt', 0)
        beta = financial_metrics.get('beta', 1.0)

        risk_free_rate = 0.04
        market_return = 0.10
        cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)

        if total_debt == 0 or market_cap == 0:
            return cost_of_equity

        total_value = market_cap + total_debt
        equity_weight = market_cap / total_value
        debt_weight = total_debt / total_value

        interest_expense = financial_metrics.get('interest_expense', 0)
        cost_of_debt = interest_expense / total_debt if total_debt > 0 else 0.05
        tax_rate = financial_metrics.get('tax_rate', 0.21)

        wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))
        return max(0.05, min(0.25, wacc))
    except Exception as e:
        logger.error(f"WACC calculation failed: {e}")
        return 0.10
