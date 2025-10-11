import streamlit as st
import pandas as pd

@st.cache_data
def load_gsheet_data():
    """
    Loads data directly from the public CSV export of the provided Google Sheet URL.
    This function replaces the complex API scraping attempt.
    """
    # Transformed URL for CSV export of the sheet with gid=1056607967
    GSHEET_EXPORT_URL = "https://docs.google.com/spreadsheets/d/15tiGXPRU1jYF_4I2ee6U3CrRBZecUgoq5N7HX_DupB4/gviz/tq?tqx=out:csv&gid=1056607967"
    # st.info(f"Attempting to load cost data from Google Sheet...")

    try:
        # Read the CSV directly into a DataFrame
        df = pd.read_csv(GSHEET_EXPORT_URL)
        df = df[['player_name', 'team', 'current_cost', 'current_selection', 'current_form', 'current_total_points']]

        # st.success(f"Successfully loaded data for {len(df)} rows from Google Sheet.")
        return df

    except Exception as e:
        st.error(
            f"Error loading data from Google Sheet. Ensure the sheet is published to the web or the URL is correct. Error: {e}")
        # Fallback to an empty DataFrame if loading fails
        return pd.DataFrame({
            'player_name': [], 'team': [], 'current_cost': [], 'current_selection': [], 'current_form': [], 'current_total_points': []
        })

