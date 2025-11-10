class MetricsExtractor:
    """
    Extracts and calculates all metrics needed for model scoring.
    """

    def __init__(self, financial_data: dict):
        self.data = financial_data

    # ---------------- Public Extraction Method ---------------- #
    def extract_all_metrics(self) -> dict:
        metrics = {}
        # Dividend metrics
        metrics['dividend_stability'] = self._calculate_dividend_stability()
        metrics['num_dividend_years'] = self._calculate_num_dividend_years()
        metrics['dividend_cagr'] = self._calculate_dividend_cagr()
        metrics['payout_ratio'] = self._calculate_payout_ratio()
        metrics['has_dividends'] = self._check_has_dividends()

        # DCF metrics
        metrics['fcf_positive_years'] = self._calculate_fcf_positive_years()
        metrics['fcf_positive_ratio'] = self._calculate_fcf_positive_ratio()
        metrics['revenue_cagr'] = self._calculate_revenue_cagr()
        metrics['has_fcf_data'] = self._check_has_fcf_data()

        # PE metrics
        metrics['eps_positive_years'] = self._calculate_eps_positive_years()
        metrics['pe_ratio'] = self._calculate_pe_ratio()
        metrics['pe_volatility'] = self._calculate_pe_volatility()
        metrics['has_pe_data'] = self._check_has_pe_data()

        # Graham metrics
        metrics['roe'] = self._calculate_roe()
        metrics['de_ratio'] = self._calculate_de_ratio()
        metrics['has_balance_sheet'] = self._check_has_balance_sheet()

        # Asset-based metrics
        metrics['asset_quality'] = self._calculate_asset_quality()
        metrics['tangible_book_ratio'] = self._calculate_tangible_book_ratio()
        metrics['has_assets_data'] = self._check_has_assets_data()

        return metrics

    # ---------------- Dividend Calculations ---------------- #
    def _calculate_dividend_stability(self):
        dividends = self.data.get('dividends', [])
        if not dividends:
            return 0
        return 1 - (sum(abs(dividends[i] - dividends[i-1]) for i in range(1, len(dividends))) / len(dividends))

    def _calculate_num_dividend_years(self):
        dividends = self.data.get('dividends', [])
        return len([d for d in dividends if d > 0])

    def _calculate_dividend_cagr(self):
        dividends = self.data.get('dividends', [])
        if len(dividends) < 2:
            return 0
        start, end = dividends[0], dividends[-1]
        n = len(dividends) - 1
        return (end / start) ** (1 / n) - 1 if start != 0 else 0

    def _calculate_payout_ratio(self):
        eps = self.data.get('eps', [])
        dividends = self.data.get('dividends', [])
        if not eps or not dividends:
            return 0
        avg_eps = sum(eps) / len(eps)
        avg_div = sum(dividends) / len(dividends)
        return avg_div / avg_eps if avg_eps != 0 else 0

    def _check_has_dividends(self):
        return int(bool(self.data.get('dividends')))

    # ---------------- DCF Calculations ---------------- #
    def _calculate_fcf_positive_years(self):
        fcf = self.data.get('fcf', [])
        return len([x for x in fcf if x > 0])

    def _calculate_fcf_positive_ratio(self):
        fcf = self.data.get('fcf', [])
        if not fcf:
            return 0
        return len([x for x in fcf if x > 0]) / len(fcf)

    def _calculate_revenue_cagr(self):
        revenue = self.data.get('revenue', [])
        if len(revenue) < 2:
            return 0
        start, end = revenue[0], revenue[-1]
        n = len(revenue) - 1
        return (end / start) ** (1 / n) - 1 if start != 0 else 0

    def _check_has_fcf_data(self):
        return int(bool(self.data.get('fcf')))

    # ---------------- PE Model Calculations ---------------- #
    def _calculate_eps_positive_years(self):
        eps = self.data.get('eps', [])
        return len([x for x in eps if x > 0])

    def _calculate_pe_ratio(self):
        price = self.data.get('price', 0)
        eps = self.data.get('eps', [])
        if not eps:
            return 0
        avg_eps = sum(eps) / len(eps)
        return price / avg_eps if avg_eps != 0 else 0

    def _calculate_pe_volatility(self):
        pe_history = self.data.get('pe_history', [])
        if not pe_history:
            return 0
        mean_pe = sum(pe_history) / len(pe_history)
        variance = sum((x - mean_pe) ** 2 for x in pe_history) / len(pe_history)
        return variance ** 0.5

    def _check_has_pe_data(self):
        return int(bool(self.data.get('eps')) and bool(self.data.get('price')))

    # ---------------- Graham Calculations ---------------- #
    def _calculate_roe(self):
        net_income = self.data.get('net_income', [])
        equity = self.data.get('equity', [])
        if not net_income or not equity:
            return 0
        avg_roe = sum([ni / eq if eq != 0 else 0 for ni, eq in zip(net_income, equity)]) / len(net_income)
        return avg_roe

    def _calculate_de_ratio(self):
        debt = self.data.get('debt', 0)
        equity_total = self.data.get('equity_total', 0)
        return debt / equity_total if equity_total != 0 else 0

    def _check_has_balance_sheet(self):
        return int(bool(self.data.get('equity_total')) and bool(self.data.get('debt')))

    # ---------------- Asset-Based Calculations ---------------- #
    def _calculate_asset_quality(self):
        tangible_assets = self.data.get('tangible_assets', 0)
        total_assets = self.data.get('total_assets', 1)
        return tangible_assets / total_assets if total_assets != 0 else 0

    def _calculate_tangible_book_ratio(self):
        tangible_assets = self.data.get('tangible_assets', 0)
        shares_outstanding = self.data.get('shares_outstanding', 1)
        return tangible_assets / shares_outstanding if shares_outstanding != 0 else 0

    def _check_has_assets_data(self):
        return int(bool(self.data.get('total_assets')) and bool(self.data.get('tangible_assets')))
