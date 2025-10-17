import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

class DataFetcher1:
    def __init__(self, ticker_symbol):
        self.ticker_symbol = ticker_symbol
        self.ticker = yf.Ticker(ticker_symbol)
        self.all_data = None
    
    def get_info(self):
        """Get basic company information"""
        return self.ticker.info
        
    def get_income_statement(self):
        """Get income statement data"""
        return self.ticker.income_stmt

    def get_balance_sheet(self):
        """Get balance sheet data"""
        return self.ticker.balance_sheet

    def get_cash_flow(self):
        """Get cash flow statement data"""
        return self.ticker.cashflow

    def get_share_price_data(self, period="max"):
        """Get historical share price data"""
        share_price_data = self.ticker.history(period=period)
        share_price_data.reset_index(inplace=True)
        return share_price_data
    
    def get_dividends(self):
        """Get dividend data"""
        return self.ticker.dividends
    
    def get_summary(self):
        """Get a summary of key metrics"""
        info = self.get_info()
        summary = {
            'company_name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'current_price': info.get('currentPrice', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'eps': info.get('trailingEps', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A')
        }
        return summary
    
    def calculate_pe_ratio(self, price, earnings_per_share):
        """Calculate P/E ratio"""
        if earnings_per_share == 0 or earnings_per_share is None or price is None:
            return None
        return price / earnings_per_share

    def get_comprehensive_data(self):
        """
        Get all financial data and combine into a comprehensive DataFrame
        
        Returns:
            pandas.DataFrame: Combined data from all sources
        """
        try:
            # Initialize data dictionary
            data_dict = {}
            
            # 1. Get basic info and summary
            info = self.get_info()
            summary = self.get_summary()
            data_dict['info'] = info
            data_dict['summary'] = summary
            
            # 2. Get financial statements
            income_stmt = self.get_income_statement()
            balance_sheet = self.get_balance_sheet()
            cash_flow = self.get_cash_flow()
            
            data_dict['income_statement'] = income_stmt
            data_dict['balance_sheet'] = balance_sheet
            data_dict['cash_flow'] = cash_flow
            
            # 3. Get market data
            share_price_data = self.get_share_price_data()
            dividends = self.get_dividends()
            
            data_dict['share_price_data'] = share_price_data
            data_dict['dividends'] = dividends
            
            # 4. Create enhanced income statement with additional metrics
            enhanced_income_stmt = self._create_enhanced_income_statement(income_stmt, share_price_data, dividends)
            data_dict['enhanced_income_statement'] = enhanced_income_stmt
            
            # 5. Combine everything into a master DataFrame
            combined_df = self._create_combined_dataframe(enhanced_income_stmt, balance_sheet, cash_flow, summary)
            
            self.all_data = data_dict
            return combined_df
            
        except Exception as e:
            print(f"Error fetching comprehensive data for {self.ticker_symbol}: {e}")
            return None
    
    def _create_enhanced_income_statement(self, income_stmt, share_price_data, dividends):
        """Create enhanced income statement with share price, dividends, and P/E ratio"""
        enhanced_is = income_stmt.copy()
        
        # Add ticker symbol
        enhanced_is.loc['Ticker'] = [self.ticker_symbol] * len(enhanced_is.columns)
        
        # Add share prices
        share_prices = []
        for date_str in enhanced_is.columns:
            fiscal_date = pd.to_datetime(date_str).date()
            mask = share_price_data['Date'].dt.date <= fiscal_date
            if mask.any():
                last_row = share_price_data[mask].iloc[-1]
                share_prices.append(last_row['Close'])
            else:
                share_prices.append(None)
        enhanced_is.loc['Share Price'] = share_prices
        
        # Add dividends
        dividend_values = []
        dividends_df = dividends.reset_index() if dividends is not None else pd.DataFrame()
        
        for date_str in enhanced_is.columns:
            fiscal_date = pd.to_datetime(date_str).date()
            if not dividends_df.empty:
                dividends_df['Date'] = pd.to_datetime(dividends_df['Date']).dt.date
                mask = dividends_df['Date'] <= fiscal_date
                if mask.any():
                    last_div = dividends_df[mask].iloc[-1]['Dividends']
                    dividend_values.append(last_div)
                else:
                    dividend_values.append(0.0)
            else:
                dividend_values.append(0.0)
        enhanced_is.loc['Dividends'] = dividend_values
        
        # Add P/E ratios
        pe_ratios = []
        for col in enhanced_is.columns:
            share_price = enhanced_is.loc["Share Price", col]
            
            # Try different EPS column names
            eps = None
            for eps_name in ["Basic EPS", "Diluted EPS", "Earnings Per Share"]:
                if eps_name in enhanced_is.index:
                    eps = enhanced_is.loc[eps_name, col]
                    break
            
            pe = self.calculate_pe_ratio(share_price, eps)
            pe_ratios.append(pe)
        enhanced_is.loc["P/E Ratio"] = pe_ratios
        
        return enhanced_is
    
    def _create_combined_dataframe(self, enhanced_income_stmt, balance_sheet, cash_flow, summary):
        """Combine all data into a single DataFrame"""
        # Start with enhanced income statement as base
        combined_df = enhanced_income_stmt.copy()
        
        # Add key balance sheet items if they exist
        balance_sheet_items = ['Total Assets', 'Total Liabilities Net Minority Interest', 
                              'Total Equity Gross Minority Interest', 'Current Assets', 
                              'Current Liabilities']
        
        for item in balance_sheet_items:
            if item in balance_sheet.index:
                combined_df.loc[item] = balance_sheet.loc[item]
        
        # Add key cash flow items if they exist
        cash_flow_items = ['Operating Cash Flow', 'Investing Cash Flow', 
                          'Financing Cash Flow', 'Free Cash Flow']
        
        for item in cash_flow_items:
            if item in cash_flow.index:
                combined_df.loc[item] = cash_flow.loc[item]
        
        # Add summary metrics as new rows
        combined_df.loc['Summary_MarketCap'] = [summary['market_cap']] * len(combined_df.columns)
        combined_df.loc['Summary_CurrentPrice'] = [summary['current_price']] * len(combined_df.columns)
        combined_df.loc['Summary_Sector'] = [summary['sector']] * len(combined_df.columns)
        combined_df.loc['Summary_Industry'] = [summary['industry']] * len(combined_df.columns)
        
        return combined_df
    
    def display_data_overview(self):
        """Display an overview of all fetched data"""
        if self.all_data is None:
            print("No data available. Run get_comprehensive_data() first.")
            return
        
        print(f"\n{'='*50}")
        print(f"DATA OVERVIEW FOR {self.ticker_symbol}")
        print(f"{'='*50}")
        
        summary = self.all_data['summary']
        print(f"Company: {summary['company_name']}")
        print(f"Sector: {summary['sector']}")
        print(f"Industry: {summary['industry']}")
        print(f"Market Cap: ${summary['market_cap']:,.2f}")
        print(f"Current Price: ${summary['current_price']:.2f}")
        print(f"P/E Ratio: {summary['pe_ratio']}")
        
        print(f"\nFinancial Statements:")
        print(f"- Income Statement: {self.all_data['income_statement'].shape}")
        print(f"- Balance Sheet: {self.all_data['balance_sheet'].shape}")
        print(f"- Cash Flow: {self.all_data['cash_flow'].shape}")
        print(f"- Share Price Data Points: {len(self.all_data['share_price_data'])}")
        print(f"- Dividend Payments: {len(self.all_data['dividends']) if self.all_data['dividends'] is not None else 0}")

# Example usage
if __name__ == "__main__":
    # Create fetcher instance
    fetcher = DataFetcher("AAPL")
    
    # Get all data in one comprehensive DataFrame
    comprehensive_df = fetcher.get_comprehensive_data()
    
    # Display overview
    fetcher.display_data_overview()
    
    # Show the combined DataFrame
    if comprehensive_df is not None:
        print(f"\nCombined DataFrame shape: {comprehensive_df.shape}")
        print("\nFirst few rows of combined data:")
        print(comprehensive_df.head())