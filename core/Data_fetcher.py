import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

class Data_fetcher:
    def __init__(self, ticker_symbol):
        self.ticker_symbol = ticker_symbol
        self.ticker = yf.Ticker(ticker_symbol)

    def get_info(self):
        return self.ticker.info
        
    def get_income_statement(self):
        return self.ticker.income_stmt

    def get_balance_sheet(self):
        return self.ticker.balance_sheet

    def get_cash_flow(self):
        return self.ticker.cashflow

    def get_share_price_data(self, period="max"):
        share_price_data = self.ticker.history(period=period)
        share_price_data.reset_index(inplace=True)
        return share_price_data
    
    # Calculate P/E Ratio row
    def calculate_pe_ratio(self, price, earnings_per_share):
        if earnings_per_share == 0 or earnings_per_share is None:
            return None
        return price / earnings_per_share

    def get_and_edit_income_statement(self):
        # Retrieve ticker data
        info = self.ticker.info
        income_statement = self.ticker.income_stmt

        # Add Ticker as the first row for easy identification
        income_statement.loc['Ticker'] = [self.ticker_symbol] * len(income_statement.columns)
        
        # Share price data
        share_price_data = self.ticker.history(period="max")
        share_price_data.reset_index(inplace=True)

        # Add Share Price row
        share_prices = []
        for date_str in income_statement.columns:
            fiscal_date = pd.to_datetime(date_str).date()
            mask = share_price_data['Date'].dt.date <= fiscal_date
            if mask.any():
                last_row = share_price_data[mask].iloc[-1]
                share_prices.append(last_row['Close'])
            else:
                share_prices.append(None)
        income_statement.loc['Share Price'] = share_prices

        dividends = self.ticker.dividends.reset_index()
        dividends['Date'] = pd.to_datetime(dividends['Date']).dt.date

        dividend_values = []
        for date_str in income_statement.columns:
            fiscal_date = pd.to_datetime(date_str).date()
            mask = dividends['Date'] <= fiscal_date
            if mask.any():
                last_div = dividends[mask].iloc[-1]['Dividends']
                dividend_values.append(last_div)
            else:
                dividend_values.append(0.0)
        income_statement.loc['Dividends'] = dividend_values

        pe_ratios = []
        for col in income_statement.columns:
            share_price = income_statement.loc["Share Price", col]
            eps = income_statement.loc["Basic EPS", col] if "Basic EPS" in income_statement.index else None
            pe = self.calculate_pe_ratio(share_price, eps)
            pe_ratios.append(pe)
        income_statement.loc["P/E Ratio"] = pe_ratios

        print("\nFinal Income Statement with P/E Ratio:")
        print(income_statement)

        return info, income_statement
    

    def fnisd():
        pass


