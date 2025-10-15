import requests
import pandas as pd
from datetime import datetime

class Ticker():
    def __init__(self, symbol):
        self.symbol = symbol

    def get_info(self):
        return f"Information about {self.symbol}"

    def get_close_price(self):
        unit_today_time = int(datetime.now().timestamp())
        url = "https://query2.finance.yahoo.com/v8/finance/chart/" + self.symbol + "?period1=946702800&period2="+  str(unit_today_time) + "&interval=1d&events=history"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            # Parse the JSON data
            data = response.json()
            # Extract the timestamp and close prices
            timestamps = data['chart']['result'][0]['timestamp']
            close_prices = data['chart']['result'][0]['indicators']['quote'][0]['close']

            # Convert to DataFrame
            df = pd.DataFrame({
                'Date': [datetime.fromtimestamp(ts) for ts in timestamps],
                'Close': close_prices
            })

            # Set the date as index
            #df.set_index('Date', inplace=True)
            return df
        
    def get_current_price(self):
        #url = f"https://query2.finance.yahoo.com/v7/finance/quote?symbols={self.symbol}"
        unit_today_time = int(datetime.now().timestamp())
        url = "https://query2.finance.yahoo.com/v8/finance/chart/" + self.symbol + "?period1=946702800&period2="+  str(unit_today_time) + "&interval=1d&events=history"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            current_price = int(data['chart']['result'][0]['meta']['regularMarketPrice'])
            return current_price
        