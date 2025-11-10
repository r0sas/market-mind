import pandas as pd
from typing import List, Dict, Optional
import warnings
import time
import concurrent.futures
import logging

logger = logging.getLogger(__name__)

class MultiCompanyAggregator:
    """
    Aggregates simplified financial data from multiple companies for sector analysis.

    Responsibilities:
    - Fetch and simplify data for multiple tickers
    - Combine into a single comparative DataFrame
    - Group by sector for analysis
    - Prepare data for visualization
    """

    def __init__(self, data_fetcher_class, simplifier_class, max_workers: int = 5, delay: float = 0.5):
        """
        Initialize with the DataFetcher and IVSimplifier classes.
        Supports concurrency with a thread pool and request delay to mitigate rate limits.

        Args:
            data_fetcher_class: DataFetcher class (not instance)
            simplifier_class: IVSimplifier class (not instance)
            max_workers: Max concurrent workers to use for fetching (default 5)
            delay: Delay (seconds) between requests to avoid rate limiting (default 0.5)
        """
        self.data_fetcher_class = data_fetcher_class
        self.simplifier_class = simplifier_class
        self.company_data: Dict[str, pd.DataFrame] = {}
        self.sector_info: Dict[str, Dict[str, any]] = {}
        self.combined_df: Optional[pd.DataFrame] = None
        self.max_workers = max_workers
        self.delay = delay

    def fetch_company_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch and simplify data for a single company, returning simplified DataFrame or None on failure.
        """
        try:
            fetcher = self.data_fetcher_class(ticker)
            comprehensive_df = fetcher.get_comprehensive_data()
            if comprehensive_df.empty:
                warnings.warn(f"Comprehensive data for {ticker} is empty.")
                return None

            summary = fetcher.get_summary()
            self.sector_info[ticker] = {
                'company_name': summary.get('company_name', 'N/A'),
                'sector': summary.get('sector', 'N/A'),
                'industry': summary.get('industry', 'N/A'),
                'market_cap': summary.get('market_cap', None)
            }

            simplifier = self.simplifier_class(comprehensive_df)
            simplified_df = simplifier.simplify()
            if simplified_df.empty:
                warnings.warn(f"Simplified data for {ticker} is empty.")
                return None

            return simplified_df
        except Exception as e:
            warnings.warn(f"Failed to fetch/simplify data for {ticker}: {e}")
            return None

    def _fetch_with_delay(self, ticker: str) -> Optional[pd.DataFrame]:
        """Helper to fetch single ticker data with delay after fetch."""
        df = self.fetch_company_data(ticker)
        time.sleep(self.delay)
        return df

    def fetch_multiple_companies(
        self,
        tickers: List[str],
        use_concurrent: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple companies.
        Supports concurrent execution with rate-limiting delay.

        Args:
            tickers: List of stock ticker symbols
            use_concurrent: Whether to fetch concurrently using ThreadPoolExecutor

        Returns:
            Dict mapping ticker symbols to simplified DataFrames
        """
        results = {}

        if use_concurrent:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Schedule fetches
                futures = {executor.submit(self.fetch_company_data, ticker): ticker for ticker in tickers}
                for future in concurrent.futures.as_completed(futures):
                    ticker = futures[future]
                    try:
                        df = future.result()
                        if df is not None:
                            results[ticker] = df
                            self.company_data[ticker] = df
                        else:
                            logger.warning(f"No data returned for {ticker}")
                    except Exception as e:
                        logger.warning(f"Exception for {ticker}: {e}")
                    finally:
                        time.sleep(self.delay)  # delay between completions to avoid rate limit
        else:
            # Sequential fetching for strict rate-limits or debugging
            for ticker in tickers:
                logger.info(f"Fetching data for {ticker}")
                df = self.fetch_company_data(ticker)
                if df is not None:
                    results[ticker] = df
                    self.company_data[ticker] = df
                else:
                    logger.warning(f"No data for {ticker}")
                time.sleep(self.delay)

        return results

    def create_combined_dataframe(
        self,
        metrics: Optional[List[str]] = None,
        latest_year_only: bool = True
    ) -> pd.DataFrame:
        """
        Combine all simplified company data into a single DataFrame.

        Args:
            metrics: List of selected metrics to include (default ?)
            latest_year_only: Whether to keep only the most recent year per company

        Returns:
            Combined pandas DataFrame with one row per company or year
        """
        if not self.company_data:
            raise ValueError("No company data to combine. Call fetch_multiple_companies first.")

        combined_data = []

        for ticker, df in self.company_data.items():
            try:
                # Remove duplicate indices and columns
                df = df.loc[~df.index.duplicated(keep='first')]
                df = df.loc[:, ~df.columns.duplicated(keep='first')]

                if df.empty:
                    logger.warning(f"{ticker}: DataFrame empty after duplicate removal, skipping.")
                    continue

                info = self.sector_info.get(ticker, {})

                if latest_year_only:
                    year_col = 'Year 1' if 'Year 1' in df.columns else df.columns[0]

                    row = {
                        "Ticker": ticker,
                        "Company": info.get("company_name", ticker),
                        "Sector": info.get("sector", "Unknown"),
                        "Industry": info.get("industry", "Unknown"),
                        "Market Cap": info.get("market_cap", None),
                        "Year": df.loc["Date", year_col] if "Date" in df.index else None
                    }

                    target_metrics = metrics if metrics else [idx for idx in df.index if idx not in ("Date", "Ticker")]
                    for m in target_metrics:
                        if m in df.index:
                            row[m] = df.loc[m, year_col]
                        else:
                            logger.debug(f"Metric '{m}' not found for {ticker}")

                    combined_data.append(row)

                else:
                    # Include all years
                    for col in df.columns:
                        year_val = df.loc["Date", col] if "Date" in df.index else col
                        row = {
                            "Ticker": ticker,
                            "Company": info.get("company_name", ticker),
                            "Sector": info.get("sector", "Unknown"),
                            "Industry": info.get("industry", "Unknown"),
                            "Market Cap": info.get("market_cap", None),
                            "Year": year_val
                        }

                        target_metrics = metrics if metrics else [idx for idx in df.index if idx not in ("Date", "Ticker")]
                        for m in target_metrics:
                            if m in df.index:
                                row[m] = df.loc[m, col]
                            else:
                                logger.debug(f"Metric '{m}' not found for {ticker}")

                        combined_data.append(row)

            except Exception as e:
                logger.warning(f"Error processing data for {ticker}: {e}")

        if not combined_data:
            raise ValueError("No valid data to combine after processing.")

        self.combined_df = pd.DataFrame(combined_data)

        # Convert numeric columns safely to numeric dtype
        numeric_cols = [
            'Market Cap', 'P/E Ratio', 'Basic EPS', 'Free Cash Flow',
            'Operating Cash Flow', 'Investing Cash Flow', 'Annual Dividends',
            'Net Income', 'Diluted EPS', 'Share Price', 'Total Assets',
            'Total Liabilities Net Minority Interest', 'Total Equity Gross Minority Interest',
            'Basic Average Shares', 'Diluted Average Shares', 'Year'
        ]

        for col in numeric_cols:
            if col in self.combined_df.columns:
                self.combined_df[col] = pd.to_numeric(self.combined_df[col], errors='coerce')

        return self.combined_df

    def get_sector_summary(self, metric: str) -> pd.DataFrame:
        """
        Get summary statistics for a metric grouped by sector.
        """
        if self.combined_df is None:
            raise ValueError("No combined data available. Call create_combined_dataframe first.")

        if metric not in self.combined_df.columns:
            raise ValueError(f"Metric '{metric}' not found in combined data.")

        sector_stats = self.combined_df.groupby('Sector')[metric].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('median', 'median'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max')
        ]).round(2)

        return sector_stats

    def get_sector_data(self, sector: str) -> pd.DataFrame:
        """Get all companies in a specific sector."""
        if self.combined_df is None:
            raise ValueError("No combined data available. Call create_combined_dataframe first.")

        return self.combined_df[self.combined_df['Sector'] == sector].copy()

    def get_top_companies(self, metric: str, n: int = 10, ascending: bool = False) -> pd.DataFrame:
        """Get top N companies by a specific metric."""
        if self.combined_df is None:
            raise ValueError("No combined data available. Call create_combined_dataframe first.")

        if metric not in self.combined_df.columns:
            raise ValueError(f"Metric '{metric}' not found in combined data.")

        return (self.combined_df.nsmallest(n, metric) if ascending else self.combined_df.nlargest(n, metric))

    def export_to_csv(self, filename: str) -> None:
        """Export combined data to CSV."""
        if self.combined_df is None:
            raise ValueError("No combined data available. Call create_combined_dataframe first.")

        self.combined_df.to_csv(filename, index=False)
        logger.info(f"Data exported to {filename}")

    def get_available_sectors(self) -> List[str]:
        """Get list of all sectors in the data."""
        if self.combined_df is None:
            raise ValueError("No combined data available. Call create_combined_dataframe first.")

        return sorted(self.combined_df['Sector'].unique().tolist())

    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics (excluding identifiers)."""
        if self.combined_df is None:
            raise ValueError("No combined data available. Call create_combined_dataframe first.")

        exclude_cols = ['Ticker', 'Company', 'Sector', 'Industry', 'Year', 'Market Cap']
        return [col for col in self.combined_df.columns if col not in exclude_cols]

    def display_summary(self) -> None:
        """Display a summary of the aggregated data."""
        print("\n" + "="*60)
        print("MULTI-COMPANY DATA SUMMARY")
        print("="*60)

        print(f"Total companies: {len(self.company_data)}")
        print(f"\nCompanies by ticker:")
        for ticker, info in self.sector_info.items():
            market_cap = info.get('market_cap')
            mc_str = f"${market_cap:,.0f}" if market_cap is not None else "N/A"
            print(f"  {ticker}: {info['company_name']} ({info['sector']}) - Market Cap: {mc_str}")

        if self.combined_df is not None:
            print(f"\nCombined DataFrame shape: {self.combined_df.shape}")
            print(f"Sectors represented: {len(self.get_available_sectors())}")
            print(f"Sectors: {', '.join(self.get_available_sectors())}")

            metrics = self.get_available_metrics()
            print(f"\nAvailable metrics ({len(metrics)}):")
            for metric in metrics:
                print(f"  - {metric}")

        print("="*60)

