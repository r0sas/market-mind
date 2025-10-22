import pandas as pd
from typing import List, Dict, Optional
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed


class MultiCompanyAggregator:
    """
    Aggregates simplified financial data from multiple companies for sector analysis.
    
    Responsibilities:
    - Fetch and simplify data for multiple tickers
    - Combine into a single comparative DataFrame
    - Group by sector for analysis
    - Prepare data for visualization
    """
    
    def __init__(self, data_fetcher_class, simplifier_class):
        """
        Initialize with the DataFetcher and IVSimplifier classes.
        
        Args:
            data_fetcher_class: DataFetcher class (not instance)
            simplifier_class: IVSimplifier class (not instance)
        """
        self.data_fetcher_class = data_fetcher_class
        self.simplifier_class = simplifier_class
        self.company_data: Dict[str, pd.DataFrame] = {}
        self.sector_info: Dict[str, Dict[str, str]] = {}
        self.combined_df: Optional[pd.DataFrame] = None
    
    def fetch_company_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch and simplify data for a single company.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Simplified DataFrame or None if fetch fails
        """
        try:
            # Fetch comprehensive data
            fetcher = self.data_fetcher_class(ticker)
            comprehensive_df = fetcher.get_comprehensive_data()
            
            # Get sector info
            summary = fetcher.get_summary()
            self.sector_info[ticker] = {
                'company_name': summary.get('company_name', 'N/A'),
                'sector': summary.get('sector', 'N/A'),
                'industry': summary.get('industry', 'N/A'),
                'market_cap': summary.get('market_cap')
            }
            
            # Simplify data
            simplifier = self.simplifier_class(comprehensive_df)
            simplified_df = simplifier.simplify()
            
            return simplified_df
            
        except Exception as e:
            warnings.warn(f"Failed to fetch data for {ticker}: {e}")
            return None
    
    def fetch_multiple_companies(
        self, 
        tickers: List[str], 
        parallel: bool = True,
        max_workers: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple companies.
        
        Args:
            tickers: List of stock ticker symbols
            parallel: Whether to fetch in parallel (default True)
            max_workers: Number of parallel workers (default 5)
            
        Returns:
            Dictionary mapping tickers to their simplified DataFrames
        """
        if parallel:
            return self._fetch_parallel(tickers, max_workers)
        else:
            return self._fetch_sequential(tickers)
    
    def _fetch_sequential(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch companies sequentially"""
        results = {}
        
        for ticker in tickers:
            print(f"Fetching {ticker}...")
            df = self.fetch_company_data(ticker)
            if df is not None:
                results[ticker] = df
                self.company_data[ticker] = df
        
        return results
    
    def _fetch_parallel(self, tickers: List[str], max_workers: int) -> Dict[str, pd.DataFrame]:
        """Fetch companies in parallel"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.fetch_company_data, ticker): ticker 
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    df = future.result()
                    if df is not None:
                        results[ticker] = df
                        self.company_data[ticker] = df
                        print(f"✓ Fetched {ticker}")
                    else:
                        print(f"✗ Failed {ticker}")
                except Exception as e:
                    warnings.warn(f"Error fetching {ticker}: {e}")
        
        return results
    
    def create_combined_dataframe(
        self, 
        metrics: Optional[List[str]] = None,
        latest_year_only: bool = True
    ) -> pd.DataFrame:
        """
        Combine all company data into a single DataFrame for comparison.
        
        Args:
            metrics: List of metrics to include (None = all common metrics)
            latest_year_only: If True, only include most recent year (default)
            
        Returns:
            Combined DataFrame with all companies
        """
        if not self.company_data:
            raise ValueError("No company data available. Call fetch_multiple_companies first.")
        
        combined_data = []
        
        for ticker, df in self.company_data.items():
            # Get company info
            company_info = self.sector_info.get(ticker, {})
            
            if latest_year_only:
                # Extract only Year 1 (most recent) data
                year_col = 'Year 1' if 'Year 1' in df.columns else df.columns[0]
                company_row = {
                    'Ticker': ticker,
                    'Company': company_info.get('company_name', ticker),
                    'Sector': company_info.get('sector', 'Unknown'),
                    'Industry': company_info.get('industry', 'Unknown'),
                    'Market Cap': company_info.get('market_cap')
                }
                
                # Get the year from the Date row if available
                if 'Date' in df.index:
                    company_row['Year'] = df.loc['Date', year_col]
                
                # Add metrics
                if metrics:
                    for metric in metrics:
                        if metric in df.index:
                            company_row[metric] = df.loc[metric, year_col]
                else:
                    # Add all metrics except Date and Ticker
                    for metric in df.index:
                        if metric not in ['Date', 'Ticker']:
                            company_row[metric] = df.loc[metric, year_col]
                
                combined_data.append(company_row)
            else:
                # Include all years (creates multiple rows per company)
                for col in df.columns:
                    year = df.loc['Date', col] if 'Date' in df.index else col
                    company_row = {
                        'Ticker': ticker,
                        'Company': company_info.get('company_name', ticker),
                        'Sector': company_info.get('sector', 'Unknown'),
                        'Industry': company_info.get('industry', 'Unknown'),
                        'Market Cap': company_info.get('market_cap'),
                        'Year': year
                    }
                    
                    # Add metrics
                    if metrics:
                        for metric in metrics:
                            if metric in df.index:
                                company_row[metric] = df.loc[metric, col]
                    else:
                        for metric in df.index:
                            if metric not in ['Date', 'Ticker']:
                                company_row[metric] = df.loc[metric, col]
                    
                    combined_data.append(company_row)
        
        self.combined_df = pd.DataFrame(combined_data)
        
        # Convert numeric columns to proper types
        numeric_cols = ['Market Cap', 'P/E Ratio', 'Basic EPS', 'Free Cash Flow', 
                       'Operating Cash Flow', 'Investing Cash Flow', 'Annual Dividends',
                       'Net Income', 'Diluted EPS', 'Share Price', 'Total Assets',
                       'Total Liabilities Net Minority Interest', 'Total Equity Gross Minority Interest',
                       'Basic Average Shares', 'Diluted Average Shares', 'Year']
        
        for col in numeric_cols:
            if col in self.combined_df.columns:
                self.combined_df[col] = pd.to_numeric(self.combined_df[col], errors='coerce')
        
        return self.combined_df
    
    def get_sector_summary(self, metric: str) -> pd.DataFrame:
        """
        Get summary statistics for a metric grouped by sector.
        
        Args:
            metric: The financial metric to analyze
            
        Returns:
            DataFrame with sector-level statistics
        """
        if self.combined_df is None:
            raise ValueError("No combined data available. Call create_combined_dataframe first.")
        
        if metric not in self.combined_df.columns:
            raise ValueError(f"Metric '{metric}' not found in combined data")
        
        # Group by sector and calculate statistics
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
        """
        Get all companies in a specific sector.
        
        Args:
            sector: Sector name
            
        Returns:
            DataFrame filtered to the specified sector
        """
        if self.combined_df is None:
            raise ValueError("No combined data available. Call create_combined_dataframe first.")
        
        return self.combined_df[self.combined_df['Sector'] == sector].copy()
    
    def get_top_companies(
        self, 
        metric: str, 
        n: int = 10, 
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Get top N companies by a specific metric.
        
        Args:
            metric: Metric to sort by
            n: Number of companies to return
            ascending: Sort order (False = highest first)
            
        Returns:
            DataFrame with top companies
        """
        if self.combined_df is None:
            raise ValueError("No combined data available. Call create_combined_dataframe first.")
        
        if metric not in self.combined_df.columns:
            raise ValueError(f"Metric '{metric}' not found in combined data")
        
        return self.combined_df.nlargest(n, metric) if not ascending else self.combined_df.nsmallest(n, metric)
    
    def export_to_csv(self, filename: str) -> None:
        """
        Export combined data to CSV.
        
        Args:
            filename: Output CSV filename
        """
        if self.combined_df is None:
            raise ValueError("No combined data available. Call create_combined_dataframe first.")
        
        self.combined_df.to_csv(filename, index=False)
        print(f"Data exported to {filename}")
    
    def get_available_sectors(self) -> List[str]:
        """Get list of all sectors in the data"""
        if self.combined_df is None:
            raise ValueError("No combined data available. Call create_combined_dataframe first.")
        
        return sorted(self.combined_df['Sector'].unique().tolist())
    
    def get_available_metrics(self) -> List[str]:
        """Get list of all available metrics in the combined data"""
        if self.combined_df is None:
            raise ValueError("No combined data available. Call create_combined_dataframe first.")
        
        exclude_cols = ['Ticker', 'Company', 'Sector', 'Industry', 'Year', 'Market Cap']
        return [col for col in self.combined_df.columns if col not in exclude_cols]
    
    def display_summary(self) -> None:
        """Display a summary of the aggregated data"""
        print("\n" + "="*60)
        print("MULTI-COMPANY DATA SUMMARY")
        print("="*60)
        
        print(f"Total companies: {len(self.company_data)}")
        print(f"\nCompanies by ticker:")
        for ticker, info in self.sector_info.items():
            market_cap = info.get('market_cap')
            mc_str = f"${market_cap:,.0f}" if market_cap else "N/A"
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


# Example usage
if __name__ == "__main__":
    from core.data_fetcher import DataFetcher
    from core.iv_simplifier import IVSimplifier
    
    # Define tickers to analyze
    tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']
    
    try:
        # Create aggregator
        aggregator = MultiCompanyAggregator(DataFetcher, IVSimplifier)
        
        # Fetch data for multiple companies
        print("Fetching data for multiple companies...")
        aggregator.fetch_multiple_companies(tech_tickers, parallel=True)
        
        # Create combined dataframe (latest year only)
        combined_df = aggregator.create_combined_dataframe(
            metrics=['Basic EPS', 'Free Cash Flow', 'P/E Ratio', 'Share Price'],
            latest_year_only=True
        )
        
        # Display summary
        aggregator.display_summary()
        
        print("\nCombined Data Preview:")
        print(combined_df[['Ticker', 'Company', 'Sector', 'Market Cap', 'P/E Ratio', 'Basic EPS']])
        
        # Get sector summary for a specific metric
        print("\n\nP/E Ratio by Sector:")
        pe_summary = aggregator.get_sector_summary('P/E Ratio')
        print(pe_summary)
        
        # Get top companies by market cap
        print("\n\nTop 5 Companies by Market Cap:")
        top_by_mc = aggregator.get_top_companies('Market Cap', n=5)
        print(top_by_mc[['Ticker', 'Company', 'Market Cap', 'P/E Ratio']])
        
        # Export to CSV
        aggregator.export_to_csv('multi_company_analysis.csv')
        
    except Exception as e:
        print(f"Error: {e}")