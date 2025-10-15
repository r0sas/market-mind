from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import date
import dash
from core.stock_portfolio import Stock_portfolio  # assume you have this somewhere

class PortfolioSection:
    def __init__(self, app, portefolio):
        self.app = app
        self.layout = self.create_layout()
        self.register_callbacks()
        self.portefolio = portefolio

    def create_layout(self):
        return html.Div([
            html.H4("Add Position"),
            dbc.Row([
                dbc.Col(dcc.Input(id='input-ticker', type='text', placeholder='Ticker', debounce=True), width=2),
                dbc.Col(dcc.Input(id='input-shares', type='number', placeholder='Shares'), width=2),
                dbc.Col(dcc.Input(id='input-price', type='number', placeholder='Price ($)', step=0.01), width=2),
                dbc.Col(dcc.DatePickerSingle(
                    id='input-date',
                    placeholder='Date',
                    display_format='DD/MM/YYYY',
                    max_date_allowed=date.today()
                ), width=3),
                dbc.Col(dbc.Button("Add", id='add-entry', color='primary', className='w-100'), width=2),
            ], className="mb-4"),

            html.H4("Your Portfolio"),
            dash_table.DataTable(
                id='portfolio-table',
                columns=[
                    {"name": "Ticker", "id": "Ticker"},
                    {"name": "Shares", "id": "Shares"},
                    {"name": "Price", "id": "Price"},
                    {"name": "Date", "id": "Date"},
                    {"name": "ROI (%)", "id": "ROI"},
                    {"name": "Average ROI (%)", "id": "Average_ROI"},
                    {"name": "Action", "id": "Action", "presentation": "markdown"},
                ],
                data=[],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center', 'padding': '5px'},
                style_header={'backgroundColor': '#1f2630', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#2b2b2b'}
                ],
            )
        ])

    def register_callbacks(self):
        # --- Add entry ---
        @self.app.callback(
            Output('portfolio-table', 'data', allow_duplicate=True),
            Input('add-entry', 'n_clicks'),
            State('input-ticker', 'value'),
            State('input-shares', 'value'),
            State('input-price', 'value'),
            State('input-date', 'date'),
            State('portfolio-table', 'data'),
            prevent_initial_call='initial_duplicate'
        )
        def add_entry(n_clicks, ticker, shares, price, entry_date, table_data):
            if not n_clicks:
                raise dash.exceptions.PreventUpdate
            elif not all([ticker, shares, price, entry_date]):
                raise dash.exceptions.PreventUpdate
            
            self.portefolio.add_position(ticker, shares, price, entry_date)

            stock_object = self.portefolio.get_stock_object(ticker)
            roi = stock_object.return_on_investment()
            df = pd.DataFrame(table_data or [])
            
            new_entry = {
                "Ticker": ticker.upper(),
                "Shares": int(shares),
                "Price": float(price),
                "Date": pd.to_datetime(entry_date).strftime("%d/%m/%Y"),
                "ROI": roi,
                "Action": f"[🗑️ Remove](remove-{len(df)})"
            }

            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

            return df.to_dict("records")

        # --- Remove entry ---
        @self.app.callback(
            Output('portfolio-table', 'data', allow_duplicate=True),
            Input('portfolio-table', 'active_cell'),
            State('portfolio-table', 'data'),
            prevent_initial_call='initial_duplicate'
        )
        def remove_entry(active_cell, table_data):
            if not active_cell or active_cell['column_id'] != 'Action':
                raise dash.exceptions.PreventUpdate
            df = pd.DataFrame(table_data)
            df = df.drop(active_cell['row']).reset_index(drop=True)

            return df.to_dict("records")

