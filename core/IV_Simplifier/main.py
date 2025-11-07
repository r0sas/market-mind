# main.py

import pandas as pd
from core.IV_Simplifier.simplify_data import IVSimplifier
from core.IV_Simplifier.display_summary import display_summary

# Example DataFrame (raw financial data)
data = {
    'Ticker': ['AAPL', 'AAPL'],
    'Date': ['2020-12-31', '2021-12-31'],
    'Revenue': [274.5, 365.8],
    'Net Income': [59.4, 85.4],
    'Free Cash Flow': [60.0, 70.0]
}
df = pd.DataFrame(data)

# Initialize the IVSimplifier
simplifier = IVSimplifier(df)

# Simplify the data
simplifier.simplify_data()

# Retrieve the simplified DataFrame
simplified_df = simplifier.get_simplified_data()
print(simplified_df)

# Display a summary of the simplified data
display_summary(simplified_df)
