import streamlit as st

class PlotSelectionDropdown:
    """Dropdown for selecting plot types."""
    def __init__(self):
        pass

    def render(self):
        plot_type = st.selectbox(
            "Select Plot Type",
            options=["Line Chart", "Bar Chart", "Scatter Plot"],
            index=0
        )
        return plot_type