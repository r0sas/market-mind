from core.Ticker import Ticker
class Stock:
    def __init__(self, symbol, shares, purchase_price, date):
        self.symbol = symbol
        self.shares = []
        self.purchase_price = []
        self.purchase_dates = []
        self.shares.append(shares)
        self.purchase_dates.append(date)
        self.purchase_price.append(purchase_price)

    def fetch_current_price(self):
        return Ticker(self.symbol).get_current_price()

    def current_value(self):
        print("Current Price:")
        print(self.fetch_current_price())
        print("Shares:")
        print(self.shares)
        return sum(self.shares) * self.fetch_current_price()
    
    def add_shares(self, shares, purchase_price, date):
        print(type(shares), type(purchase_price))
        self.shares.append(int(shares))
        self.purchase_price.append(purchase_price)
        self.purchase_dates.append(date)
    
    def get_symbol(self):
        return self.symbol

    def return_on_investment(self):
        total_invested = sum([s * p for s, p in zip(self.shares, self.purchase_price)])
        current_value = self.current_value()
        if total_invested == 0:
            return 0
        return ((current_value - total_invested) / total_invested) * 100

    def profit_loss(self, current_price):
        return (current_price - self.purchase_price) * self.shares