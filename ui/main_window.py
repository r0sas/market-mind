from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px
import pandas as pd
from core.Ticker import Ticker
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeSwitchAIO
from datetime import date

class MainWindow:
    def __init__(self):
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME])
        self.theme_switch = ThemeSwitchAIO(aio_id="theme", themes=[dbc.themes.BOOTSTRAP, dbc.themes.DARKLY])
        self.app.layout = dbc.Container([
        self.theme_switch,
        html.H2("Stock Dashboard"),

        # Text input
        dcc.Input(
            id='text-input',
            type='text',
            placeholder='Enter ticker...',
            debounce=True,
            style={'width': '200px', 'height': '30px'}),
        # Date picker
        dcc.DatePickerRange(
            id='my-date-picker-single',
            start_date_placeholder_text="Start Period",
            end_date_placeholder_text="End Period",
            calendar_orientation='horizontal',
            initial_visible_month=date.today(),
            max_date_allowed=date.today(),
            style={'height': '20px'}),
        # Dropdown checklist container
        html.Div(
            children=dbc.DropdownMenu(
                children=[
                    dcc.Checklist(
                        id='ticker-checklist',
                        options=[],
                        value=[],
                        inline=False
                    ),
                ],
                label="Tickers",
                id='ticker-dropdown',
                color="secondary",
                className="mt-3"),),
        # Add button
        dbc.Button("Add Ticker", id="add-button", color="primary", className="mt-2"),
        html.Div(id='output-text', className="mt-3"),
        dcc.Graph(id='output-graph', figure=px.scatter(template='darkly'))
        ])
        self.register_callbacks()

    def register_callbacks(self):
        # --- Callback: Add ticker to dropdown checklist ---
        @self.app.callback(
            Output('ticker-checklist', 'options'),
            Output('ticker-checklist', 'value'),
            Input('add-button', 'n_clicks'),
            State('text-input', 'value'),
            State('ticker-checklist', 'options'),
            State('ticker-checklist', 'value'),
            prevent_initial_call=True
        )
        def add_ticker(n_clicks, ticker, options, values):
            print(ticker)
            if not ticker or ticker.strip() == "":
                return options, values

            ticker = ticker.upper()
            existing = [opt["value"] for opt in options]
            if ticker not in existing:
                options.append({"label": ticker, "value": ticker})
                values.append(ticker)

            return options, values
        # --- Callback: Update graph based on selected tickers ---
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

            fig = px.line(combined_df, x='Date', y='Close', color='Ticker',
                        title=f"Price Data for {', '.join(selected_tickers)}", template='darkly')
            return f"Showing {len(selected_tickers)} tickers.", fig


    def run(self):
        self.app.run(debug=True)
