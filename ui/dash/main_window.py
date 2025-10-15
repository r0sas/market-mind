from dash import Dash, html, dcc, Input, Output, State, dash_table
import plotly.express as px
import pandas as pd
from core.Ticker import Ticker
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeSwitchAIO
from datetime import date
from ui.components.controls_section import ControlsSection
from ui.components.graph_section import GraphSection
from ui.components.portfolio_section import PortfolioSection
import dash
from dash.exceptions import PreventUpdate

class MainWindow:
    # def __init__(self):
    #     # Initialize the Dash app
    #     self.app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME])
    #     self.theme_switch = ThemeSwitchAIO(aio_id="theme", themes=[dbc.themes.BOOTSTRAP, dbc.themes.DARKLY])

    #     # Instantiate sections
    #     self.controls = ControlsSection(self.app)
    #     self.graphs = GraphSection(self.app)

    #     # Build full layout
    #     self.app.layout = dbc.Container([
    #         self.theme_switch,
    #         html.H2("Stock Dashboard", className="mt-3"),
    #         self.controls.layout,
    #         html.Hr(),
    #         self.graphs.layout
    #     ])

    # def run(self):
    #     self.app.run(debug=True)

    def __init__(self, portfolio):
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME])
        self.theme_switch = ThemeSwitchAIO(aio_id="theme", themes=[dbc.themes.BOOTSTRAP, dbc.themes.DARKLY])
        self.portfolio = portfolio
        # Create the portfolio section
        self.portfolio_section = PortfolioSection(self.app, self.portfolio)

        # Main layout
        self.app.layout = dbc.Container([
            self.theme_switch,
            html.H2("Portfolio Tracker", className="mt-3 mb-4 text-center"),
            self.portfolio_section.layout
        ])

    def run(self):
        self.app.run(debug=True)