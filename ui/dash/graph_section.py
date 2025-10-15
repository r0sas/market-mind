from dash import html, dcc, Input, Output
import plotly.express as px
import pandas as pd
from core.Ticker import Ticker  # assume you have this somewhere

class GraphSection:
    def __init__(self, app):
        self.app = app
        self.layout = self.create_layout()
        self.register_callbacks()

    def create_layout(self):
        return html.Div([
            html.Div(id='output-text', className="mt-3"),
            dcc.Graph(id='output-graph', figure=px.scatter(template='darkly'))
        ])

    def register_callbacks(self):
        @self.app.callback(
            [Output('output-text', 'children'),
             Output('output-graph', 'figure')],
            Input('ticker-checklist', 'value')
        )
        def update_graph(selected_tickers):
            if not selected_tickers:
                return "No tickers selected.", px.scatter(template='darkly')

            combined_df = pd.DataFrame()
            for ticker in selected_tickers:
                try:
                    df = Ticker(ticker).get_close_price()
                    df["Ticker"] = ticker
                    combined_df = pd.concat([combined_df, df])
                except Exception:
                    continue

            if combined_df.empty:
                return "No valid data found.", px.scatter(template='darkly')

            fig = px.line(
                combined_df,
                x='Date',
                y='Close',
                color='Ticker',
                title=f"Price Data for {', '.join(selected_tickers)}",
                template='darkly'
            )
            return f"Showing {len(selected_tickers)} tickers.", fig