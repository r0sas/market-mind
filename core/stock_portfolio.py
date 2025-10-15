from core.stock import Stock  # assume you have this somewhere
from core.Ticker import Ticker
import pandas as pd

class Stock_portfolio:
    def __init__(self):
        self.stocks_symbols = []
        self.stocks = []

    def add_position(self, stock_symbol, shares, purchase_price, date):
        if stock_symbol not in self.stocks_symbols:
            self.stocks_symbols.append(stock_symbol)
            self.stocks.append(Stock(stock_symbol, shares, purchase_price, date))
            # existing_stock = self.stocks[stock.symbol]
            # existing_stock.shares.append(stock.shares[0])
            # existing_stock.purchase_price.append(stock.purchase_price[0])
        else:
            index = self.stocks_symbols.index(stock_symbol)
            stock_object = self.stocks[index]
            stock_object.add_shares(shares, purchase_price, date)

    def get_stock_object(self, stock_symbol):
        if stock_symbol in self.stocks_symbols:
            index = self.stocks_symbols.index(stock_symbol)
            return self.stocks[index]
        return None
    
    def get_historical_data(self, selected_stocks):
        df = pd.DataFrame(columns=[
        "Date", "Close", "Ticker"])
        for symbol in selected_stocks:
            ticker = Ticker(symbol)
            hist_data = ticker.get_close_price()
            hist_data["Ticker"] = symbol
            df = pd.concat([df, hist_data], ignore_index=True)
        return df

    # def total_current_value(self, price_lookup):
    #     total_value = 0
    #     for stock in self.stocks.values():
    #         current_price = price_lookup(stock.symbol)
    #         total_value += stock.current_value(current_price)
    #     return total_value

    # def total_invested(self):
    #     total_invested = 0
    #     for stock in self.stocks.values():
    #         total_invested += sum([s * p for s, p in zip(stock.shares, stock.purchase_price)])
    #     return total_invested

    # def overall_return_on_investment(self, price_lookup):
    #     total_invested = self.total_invested()
    #     total_current_value = self.total_current_value(price_lookup)
    #     if total_invested == 0:
    #         return 0
    #     return ((total_current_value - total_invested) / total_invested) * 100