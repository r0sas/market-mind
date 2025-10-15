from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import date

class ControlsSection:
    def __init__(self, app):
        self.app = app
        self.layout = self.create_layout()
        self.register_callbacks()

    def create_layout(self):
        return html.Div([
            dcc.Input(
                id='text-input',
                type='text',
                placeholder='Enter ticker...',
                debounce=True,
                style={'width': '200px', 'height': '30px'}
            ),

            dcc.DatePickerRange(
                id='date-picker',
                start_date_placeholder_text="Start Period",
                end_date_placeholder_text="End Period",
                initial_visible_month=date.today(),
                max_date_allowed=date.today(),
                calendar_orientation='horizontal'
            ),

            html.Div(
                dbc.DropdownMenu(
                    children=[dcc.Checklist(id='ticker-checklist', options=[], value=[], inline=False)],
                    label="Tickers",
                    id='ticker-dropdown',
                    color="secondary",
                    className="mt-3"
                )
            ),

            dbc.Button("Add Ticker", id="add-button", color="primary", className="mt-2"),
        ])

    def register_callbacks(self):
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
            if not ticker or ticker.strip() == "":
                return options, values

            ticker = ticker.upper()
            existing = [opt["value"] for opt in options]
            if ticker not in existing:
                options.append({"label": ticker, "value": ticker})
                values.append(ticker)
            return options, values