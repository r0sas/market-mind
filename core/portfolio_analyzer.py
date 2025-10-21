from core.stock_calculator import StockCalculator
import pandas as pd
import numpy as np
from typing import Optional, Dict
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# ==============================================================================
# LAYER 3: PORTFOLIO ANALYZER - Portfolio-Level Operations
# ==============================================================================

class PortfolioAnalyzer:
    """
    Manages portfolio-level operations and orchestrates analysis workflows.
    
    This is the main interface for portfolio analysis. It coordinates:
    - Building portfolio timelines from transactions
    - Aggregating individual positions into portfolio metrics
    - Computing sector and industry allocations
    - Providing portfolio-level analytics
    
    Delegates stock-level calculations to StockCalculator.
    """
    
    def __init__(self, calculator: Optional[StockCalculator] = None):
        """
        Initialize the PortfolioAnalyzer with a stock calculator.
        
        Args:
            calculator: Optional StockCalculator instance. Creates new one if not provided.
        """
        self._calculator = calculator or StockCalculator()
    
    # -------------------
    # Input Validation
    # -------------------
    
    def _validate_transactions(self, df: pd.DataFrame) -> None:
        """
        Validate transactions DataFrame has required structure.
        
        Args:
            df: Transactions DataFrame to validate
            
        Raises:
            ValueError: If DataFrame is invalid
        """
        if df.empty:
            raise ValueError("Transactions DataFrame cannot be empty")
        
        required_columns = ['Ticker', 'Date', 'Shares', 'Price']
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Check for negative values
        if (df['Shares'] < 0).any():
            raise ValueError("Shares cannot be negative")
        if (df['Price'] < 0).any():
            raise ValueError("Price cannot be negative")
        
        # Validate dates
        try:
            pd.to_datetime(df['Date'])
        except Exception as e:
            raise ValueError(f"Invalid date format in transactions: {str(e)}")
        
        # Check for empty tickers
        if df['Ticker'].isna().any() or (df['Ticker'] == '').any():
            raise ValueError("Ticker symbols cannot be empty or NaN")
        
        logger.info(f"Validated {len(df)} transactions for {df['Ticker'].nunique()} tickers")
    
    # -------------------
    # Main Analysis Pipeline
    # -------------------
    
    def analyze_portfolio(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive portfolio analysis combining individual holdings and aggregate metrics.
        
        This is the primary entry point for retrieving complete portfolio analytics.
        Orchestrates the full pipeline to produce a dataset ready for charting and reporting.
        
        Args:
            transactions_df: DataFrame with columns ['Ticker', 'Date', 'Shares', 'Price']
                            representing all purchase transactions
            
        Returns:
            Combined DataFrame with individual ticker rows and portfolio-level aggregated row,
            indexed by date, including all performance metrics and rentability calculations
            
        Raises:
            ValueError: If transactions_df is invalid
            
        Example:
            >>> transactions = pd.DataFrame({
            ...     'Ticker': ['AAPL', 'MSFT'],
            ...     'Date': ['2024-01-01', '2024-01-15'],
            ...     'Shares': [10, 5],
            ...     'Price': [150.0, 350.0]
            ... })
            >>> analyzer = PortfolioAnalyzer()
            >>> result = analyzer.analyze_portfolio(transactions)
        """
        print("transactions_df")
        print(transactions_df)
        self._validate_transactions(transactions_df)
        
        logger.info("Starting portfolio analysis")
        # Build complete portfolio timeline with all metrics
        portfolio_timeline = self.build_portfolio_timeline(transactions_df)
        logger.info(f"Portfolio analysis complete: {len(portfolio_timeline)} rows generated")
        
        return portfolio_timeline
    
    def build_portfolio_timeline(
        self, 
        transactions_df: pd.DataFrame, 
        start_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Build complete portfolio timeline from transactions and market data.
        
        Constructs daily portfolio state by merging purchase transactions with historical
        price data, computing cumulative positions, current valuations, returns, and 
        dividends. Includes both individual position and portfolio-level aggregation.
        
        Args:
            transactions_df: DataFrame with columns ['Ticker', 'Date', 'Shares', 'Price']
                            representing all purchase transactions
            start_date: Optional override for data fetch start date (defaults to first transaction)
            
        Returns:
            DataFrame indexed by Date with both individual ticker rows and 'Portfolio' aggregated row,
            including cumulative shares, investments, valuations, returns, and dividends
            
        Raises:
            ValueError: If transactions_df is invalid or data fetching fails
        """
        transaction_start_date = start_date or transactions_df["Date"].min()
        unique_tickers = transactions_df["Ticker"].unique().tolist()
        
        logger.info(f"Building timeline from {transaction_start_date} for {len(unique_tickers)} tickers")
        
        # Normalize transaction dates to midnight for consistent merging
        normalized_transactions = transactions_df.copy()
        normalized_transactions['Date'] = pd.to_datetime(
            normalized_transactions['Date']
        ).dt.tz_localize(None).dt.normalize()
        
        # Fetch historical price data from first transaction onwards
        market_history = self._calculator.get_historical_data(
            unique_tickers, 
            start_date=transaction_start_date
        )
        
        # Orchestrated pipeline: merge → accumulate → valuate → dividends → rentability → aggregate
        portfolio_timeline = (
            self._merge_transactions_with_prices(market_history, normalized_transactions)
            .pipe(self._calculator.compute_cumulative_holdings)
            .pipe(self._calculator.compute_position_valuation)
            .pipe(self._calculator.compute_accumulated_dividends)
            .pipe(self._calculator.compute_rentability)
            .pipe(self._aggregate_portfolio_metrics)
        )
        
        return portfolio_timeline
    
    # -------------------
    # Timeline Construction Helpers
    # -------------------
    
    def _merge_transactions_with_prices(
        self, 
        price_history: pd.DataFrame, 
        transactions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge historical price data with transaction records.
        
        Args:
            price_history: Historical OHLCV data from market data provider
            transactions: Purchase transaction records with shares and cost basis
            
        Returns:
            Merged DataFrame with both price data and transaction records
        """
        merged = price_history.merge(
            transactions[['Ticker', 'Date', 'Shares', 'Investment ($)']],
            on=['Ticker', 'Date'],
            how='left'
        ).sort_values(['Ticker', 'Date'])
        
        logger.info(f"Merged transactions with price history: {len(merged)} rows")
        return merged
    
    def _aggregate_portfolio_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add portfolio-level aggregated metrics to position-level data.
        
        Computes daily portfolio totals (value, investment, gains, returns, rentability)
        and appends them as a synthetic 'Portfolio' ticker row for each date.
        
        Args:
            df: DataFrame with individual position metrics and rentability
            
        Returns:
            DataFrame with added portfolio-level rows (Ticker='Portfolio') for each date
        """
        # Aggregate daily metrics across all positions
        daily_totals = df.groupby('Date').agg({
            'Current Value': 'sum',
            'Investment ($)': 'sum',
            'dividends': 'sum',
            'Cumulative Dividends': 'sum',
            'Rentability': 'mean'
        }).reset_index()
        
        # Identify as portfolio row and fill metadata columns
        daily_totals['Ticker'] = 'Portfolio'
        daily_totals['close'] = 0.0
        daily_totals['volume'] = 0.0
        daily_totals['Shares'] = 0.0
        
        # Calculate portfolio-level performance metrics using calculator
        daily_totals = self._calculator.compute_portfolio_metrics(daily_totals)
        
        # Combine with individual positions
        combined = pd.concat([df, daily_totals], ignore_index=True)
        
        logger.info(f"Added portfolio aggregation: {len(daily_totals)} portfolio rows")
        return combined
    
    # -------------------
    # Allocation Analysis
    # -------------------
    
    def get_sector_allocation(
        self, 
        portfolio_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate portfolio allocation by sector over time.
        
        Automatically fetches sector information for all tickers in the portfolio
        and computes the percentage of portfolio value invested in each sector on each date.
        
        Args:
            portfolio_data: Output from analyze_portfolio() containing timeline with Current Value
            
        Returns:
            DataFrame with columns [Date, Sector, Current Value, Portfolio Weight (%)],
            sorted by date and sector
            
        Example:
            >>> analyzer = PortfolioAnalyzer()
            >>> portfolio = analyzer.analyze_portfolio(transactions)
            >>> sector_alloc = analyzer.get_sector_allocation(portfolio)
        """
        # Filter to individual stocks (exclude 'Portfolio' row)
        stocks_only = portfolio_data[portfolio_data['Ticker'] != 'Portfolio'].copy()
        
        if stocks_only.empty:
            logger.warning("No individual stock data found for sector allocation")
            return pd.DataFrame(columns=['Date', 'Sector', 'Current Value', 'Portfolio Weight (%)'])
        
        # Get unique tickers and fetch their sector information
        unique_tickers = stocks_only['Ticker'].unique().tolist()
        stock_info = self._calculator.get_stock_info(unique_tickers)
        
        # Create sector mapping from fetched data
        sector_mapping = {
            ticker: info.get('sector', 'Unknown') 
            for ticker, info in stock_info.items()
        }
        
        # Map sectors to stocks
        stocks_only['Sector'] = stocks_only['Ticker'].map(sector_mapping)
        
        # Aggregate by sector and date
        sector_allocation = stocks_only.groupby(['Date', 'Sector']).agg({
            'Current Value': 'sum'
        }).reset_index()
        
        # Calculate portfolio weight percentage
        total_portfolio_value = stocks_only.groupby('Date')['Current Value'].sum().reset_index()
        total_portfolio_value.rename(columns={'Current Value': 'Total Value'}, inplace=True)
        
        sector_allocation = sector_allocation.merge(total_portfolio_value, on='Date')
        sector_allocation['Portfolio Weight (%)'] = (
            (sector_allocation['Current Value'] / sector_allocation['Total Value']) * 100
        )
        
        result = sector_allocation[['Date', 'Sector', 'Current Value', 'Portfolio Weight (%)']].sort_values(['Date', 'Sector'])
        logger.info(f"Calculated sector allocation: {sector_allocation['Sector'].nunique()} sectors")
        
        return result
    
    def get_industry_allocation(
        self, 
        portfolio_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate portfolio allocation by industry over time.
        
        Automatically fetches industry information for all tickers in the portfolio
        and computes the percentage of portfolio value invested in each industry on each date.
        
        Args:
            portfolio_data: Output from analyze_portfolio() containing timeline with Current Value
            
        Returns:
            DataFrame with columns [Date, Industry, Current Value, Portfolio Weight (%)],
            sorted by date and industry
            
        Example:
            >>> analyzer = PortfolioAnalyzer()
            >>> portfolio = analyzer.analyze_portfolio(transactions)
            >>> industry_alloc = analyzer.get_industry_allocation(portfolio)
        """
        # Filter to individual stocks (exclude 'Portfolio' row)
        stocks_only = portfolio_data[portfolio_data['Ticker'] != 'Portfolio'].copy()
        
        if stocks_only.empty:
            logger.warning("No individual stock data found for industry allocation")
            return pd.DataFrame(columns=['Date', 'Industry', 'Current Value', 'Portfolio Weight (%)'])
        
        # Get unique tickers and fetch their industry information
        unique_tickers = stocks_only['Ticker'].unique().tolist()
        stock_info = self._calculator.get_stock_info(unique_tickers)
        
        # Create industry mapping from fetched data
        industry_mapping = {
            ticker: info.get('industry', 'Unknown') 
            for ticker, info in stock_info.items()
        }
        
        # Map industries to stocks
        stocks_only['Industry'] = stocks_only['Ticker'].map(industry_mapping)
        
        # Aggregate by industry and date
        industry_allocation = stocks_only.groupby(['Date', 'Industry']).agg({
            'Current Value': 'sum'
        }).reset_index()
        
        # Calculate portfolio weight percentage
        total_portfolio_value = stocks_only.groupby('Date')['Current Value'].sum().reset_index()
        total_portfolio_value.rename(columns={'Current Value': 'Total Value'}, inplace=True)
        
        industry_allocation = industry_allocation.merge(total_portfolio_value, on='Date')
        industry_allocation['Portfolio Weight (%)'] = (
            (industry_allocation['Current Value'] / industry_allocation['Total Value']) * 100
        )
        
        result = industry_allocation[['Date', 'Industry', 'Current Value', 'Portfolio Weight (%)']].sort_values(['Date', 'Industry'])
        logger.info(f"Calculated industry allocation: {industry_allocation['Industry'].nunique()} industries")
        
        return result
    
    def get_latest_allocations(
        self, 
        portfolio_data: pd.DataFrame, 
        include_sector: bool = True,
        include_industry: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Get current (latest date) sector and industry allocations.
        
        Automatically fetches sector and industry information for all tickers
        and returns the most recent portfolio breakdown.
        
        Args:
            portfolio_data: Output from analyze_portfolio()
            include_sector: Whether to include sector allocation (default: True)
            include_industry: Whether to include industry allocation (default: True)
            
        Returns:
            Dictionary with keys 'sector' and/or 'industry', each containing the latest
            allocation DataFrame filtered to the most recent date
            
        Example:
            >>> analyzer = PortfolioAnalyzer()
            >>> portfolio = analyzer.analyze_portfolio(transactions)
            >>> latest = analyzer.get_latest_allocations(portfolio)
            >>> print(latest['sector'])
        """
        if portfolio_data.empty:
            logger.warning("Empty portfolio data provided")
            return {}
        
        latest_date = portfolio_data['Date'].max()
        allocations = {}
        
        logger.info(f"Getting latest allocations for date: {latest_date}")
        
        if include_sector:
            sector_alloc = self.get_sector_allocation(portfolio_data)
            if not sector_alloc.empty:
                allocations['sector'] = sector_alloc[sector_alloc['Date'] == latest_date]
        
        if include_industry:
            industry_alloc = self.get_industry_allocation(portfolio_data)
            if not industry_alloc.empty:
                allocations['industry'] = industry_alloc[industry_alloc['Date'] == latest_date]
        
        return allocations
    
    # -------------------
    # Summary Statistics
    # -------------------
    
    def get_portfolio_summary(self, portfolio_data: pd.DataFrame) -> Dict:
        """
        Generate comprehensive portfolio summary statistics.
        
        Args:
            portfolio_data: Output from analyze_portfolio()
            
        Returns:
            Dictionary containing key portfolio metrics
            
        Example:
            >>> analyzer = PortfolioAnalyzer()
            >>> portfolio = analyzer.analyze_portfolio(transactions)
            >>> summary = analyzer.get_portfolio_summary(portfolio)
            >>> print(f"Total Return: {summary['total_return_pct']:.2f}%")
        """
        if portfolio_data.empty:
            logger.warning("Empty portfolio data provided for summary")
            return {}
        
        # Get latest portfolio row
        portfolio_rows = portfolio_data[portfolio_data['Ticker'] == 'Portfolio']
        if portfolio_rows.empty:
            logger.warning("No portfolio aggregation found in data")
            return {}
        
        latest = portfolio_rows.iloc[-1]
        first = portfolio_rows.iloc[0]
        
        summary = {
            'current_value': float(latest['Current Value']),
            'total_investment': float(latest['Investment ($)']),
            'total_profit_loss': float(latest['Profit/Loss']),
            'total_return_pct': float(latest['ROI (%)']),
            'cumulative_rentability_pct': float(latest.get('Cumulative Rentability', 0)),
            'total_dividends': float(latest.get('Cumulative Dividends', 0)),
            'start_date': str(first['Date'].date()) if pd.notna(first['Date']) else None,
            'end_date': str(latest['Date'].date()) if pd.notna(latest['Date']) else None,
            'num_positions': int(portfolio_data[portfolio_data['Ticker'] != 'Portfolio']['Ticker'].nunique()),
            'days_invested': int((latest['Date'] - first['Date']).days) if pd.notna(latest['Date']) else 0
        }
        
        # Calculate annualized return if we have enough data
        if summary['days_invested'] > 0:
            years = summary['days_invested'] / 365.25
            if years < 1:
                years = 1
            summary['annualized_return_pct'] = (
                ((summary['current_value'] / summary['total_investment']) ** (1 / years) - 1) * 100
                if summary['total_investment'] > 0 else 0
            )
        else:
            summary['annualized_return_pct'] = 0
        
        logger.info(f"Generated portfolio summary: ${summary['current_value']:,.2f} current value")
        return summary
    
    def get_position_summary(self, portfolio_data: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary of all current positions with latest metrics.
        
        Args:
            portfolio_data: Output from analyze_portfolio()
            
        Returns:
            DataFrame with one row per position showing current metrics
            
        Example:
            >>> analyzer = PortfolioAnalyzer()
            >>> portfolio = analyzer.analyze_portfolio(transactions)
            >>> positions = analyzer.get_position_summary(portfolio)
        """
        if portfolio_data.empty:
            logger.warning("Empty portfolio data provided for position summary")
            return pd.DataFrame()
        
        # Get latest date data for each ticker (excluding Portfolio)
        latest_date = portfolio_data['Date'].max()
        latest_positions = portfolio_data[
            (portfolio_data['Date'] == latest_date) & 
            (portfolio_data['Ticker'] != 'Portfolio')
        ].copy()
        
        if latest_positions.empty:
            logger.warning("No position data found for latest date")
            return pd.DataFrame()
        
        # Select relevant columns for summary
        summary_cols = [
            'Ticker', 'Shares', 'Investment ($)', 'Current Value', 
            'Profit/Loss', 'ROI (%)', 'Cumulative Dividends'
        ]
        
        # Filter to columns that exist
        available_cols = [col for col in summary_cols if col in latest_positions.columns]
        position_summary = latest_positions[available_cols].copy()
        
        # Sort by current value descending
        if 'Current Value' in position_summary.columns:
            position_summary = position_summary.sort_values('Current Value', ascending=False)
        
        logger.info(f"Generated position summary for {len(position_summary)} positions")
        return position_summary.reset_index(drop=True)
    
    # -------------------
    # Performance Analysis
    # -------------------
    
    def calculate_volatility(
        self, 
        portfolio_data: pd.DataFrame, 
        ticker: Optional[str] = 'Portfolio',
        window: int = 30
    ) -> pd.DataFrame:
        """
        Calculate rolling volatility (standard deviation of returns).
        
        Args:
            portfolio_data: Output from analyze_portfolio()
            ticker: Ticker to analyze (default: 'Portfolio' for portfolio-level)
            window: Rolling window in days (default: 30)
            
        Returns:
            DataFrame with Date and Volatility columns
            
        Example:
            >>> analyzer = PortfolioAnalyzer()
            >>> portfolio = analyzer.analyze_portfolio(transactions)
            >>> volatility = analyzer.calculate_volatility(portfolio, window=60)
        """
        ticker_data = portfolio_data[portfolio_data['Ticker'] == ticker].copy()
        
        if ticker_data.empty:
            logger.warning(f"No data found for ticker: {ticker}")
            return pd.DataFrame()
        
        # Calculate rolling volatility of rentability
        if 'Rentability' not in ticker_data.columns:
            logger.warning("Rentability column not found")
            return pd.DataFrame()
        
        ticker_data = ticker_data.sort_values('Date')
        ticker_data['Volatility'] = ticker_data['Rentability'].rolling(window=window).std() * 100
        
        result = ticker_data[['Date', 'Volatility']].dropna()
        logger.info(f"Calculated volatility for {ticker} with {window}-day window")
        
        return result
    
    def calculate_sharpe_ratio(
        self,
        portfolio_data: pd.DataFrame,
        risk_free_rate: float = 0.04,
        ticker: Optional[str] = 'Portfolio'
    ) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Args:
            portfolio_data: Output from analyze_portfolio()
            risk_free_rate: Annual risk-free rate (default: 0.04 for 4%)
            ticker: Ticker to analyze (default: 'Portfolio' for portfolio-level)
            
        Returns:
            Sharpe ratio value
            
        Example:
            >>> analyzer = PortfolioAnalyzer()
            >>> portfolio = analyzer.analyze_portfolio(transactions)
            >>> sharpe = analyzer.calculate_sharpe_ratio(portfolio, risk_free_rate=0.05)
        """
        ticker_data = portfolio_data[portfolio_data['Ticker'] == ticker].copy()
        
        if ticker_data.empty or 'Rentability' not in ticker_data.columns:
            logger.warning(f"Cannot calculate Sharpe ratio for {ticker}")
            return 0.0
        
        # Get daily returns
        daily_returns = ticker_data['Rentability'].dropna()
        
        if len(daily_returns) == 0:
            return 0.0
        
        # Calculate excess returns
        daily_risk_free = risk_free_rate / 252  # Assuming 252 trading days
        excess_returns = daily_returns - daily_risk_free
        
        # Annualized Sharpe ratio
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        
        logger.info(f"Calculated Sharpe ratio for {ticker}: {sharpe:.4f}")
        return float(sharpe)
    
    def calculate_max_drawdown(
        self,
        portfolio_data: pd.DataFrame,
        ticker: Optional[str] = 'Portfolio'
    ) -> Dict:
        """
        Calculate maximum drawdown (largest peak-to-trough decline).
        
        Args:
            portfolio_data: Output from analyze_portfolio()
            ticker: Ticker to analyze (default: 'Portfolio' for portfolio-level)
            
        Returns:
            Dictionary with max_drawdown_pct, peak_date, trough_date, recovery_date
            
        Example:
            >>> analyzer = PortfolioAnalyzer()
            >>> portfolio = analyzer.analyze_portfolio(transactions)
            >>> drawdown = analyzer.calculate_max_drawdown(portfolio)
            >>> print(f"Max Drawdown: {drawdown['max_drawdown_pct']:.2f}%")
        """
        ticker_data = portfolio_data[portfolio_data['Ticker'] == ticker].copy()
        
        if ticker_data.empty or 'Current Value' not in ticker_data.columns:
            logger.warning(f"Cannot calculate max drawdown for {ticker}")
            return {'max_drawdown_pct': 0, 'peak_date': None, 'trough_date': None, 'recovery_date': None}
        
        ticker_data = ticker_data.sort_values('Date')
        values = ticker_data['Current Value'].values
        dates = ticker_data['Date'].values
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(values)
        
        # Calculate drawdown
        drawdown = (values - running_max) / running_max * 100
        
        # Find maximum drawdown
        max_dd_idx = np.argmin(drawdown)
        max_drawdown_pct = float(drawdown[max_dd_idx])
        
        # Find peak before max drawdown
        peak_idx = np.argmax(running_max[:max_dd_idx + 1] == values[:max_dd_idx + 1])
        
        # Find recovery date (if any)
        recovery_idx = None
        for i in range(max_dd_idx + 1, len(values)):
            if values[i] >= values[peak_idx]:
                recovery_idx = i
                break
        
        result = {
            'max_drawdown_pct': max_drawdown_pct,
            'peak_date': str(pd.Timestamp(dates[peak_idx]).date()),
            'trough_date': str(pd.Timestamp(dates[max_dd_idx]).date()),
            'recovery_date': str(pd.Timestamp(dates[recovery_idx]).date()) if recovery_idx is not None else None
        }
        
        logger.info(f"Max drawdown for {ticker}: {max_drawdown_pct:.2f}%")
        return result