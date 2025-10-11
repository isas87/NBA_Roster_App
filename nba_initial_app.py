import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO

# Import the data processing functions from the separate file
from data_load_processing import scrape_current_data, calculate_z_scores


# --- 2. Comparison Plotting Function (Unchanged) ---

def plot_z_score_comparison(df_rankings, players):
    """
    Generates a Matplotlib bar chart comparing the Composite Z-Scores of selected players.
    """
    if df_rankings.empty or not players:
        return None

    df_plot = df_rankings.loc[df_rankings.index.intersection(players), ['Composite_Z_Score']].sort_values(
        'Composite_Z_Score', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['#DB342A' if score == df_plot['Composite_Z_Score'].max() else '#3498DB' for score in
              df_plot['Composite_Z_Score']]

    ax.bar(df_plot.index, df_plot['Composite_Z_Score'], color=colors)

    ax.set_title('Composite Z-Score Comparison', fontsize=16)
    ax.set_xlabel('Player', fontsize=12)
    ax.set_ylabel('Composite Z-Score', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()

    return fig


# --- 3. Streamlit Application Layout ---

# Configuration and Initialization
st.set_page_config(layout="wide", page_title="NBA Fantasy Analyzer - Live Data")

# Call the imported data loading function
df_raw = scrape_current_data()
FANTASY_STATS = ['pts', 'trb', 'ast', 'stl', 'blk', 'pts_fg']

st.title("🏀 NBA Fantasy Roster Analyzer (Live Data)")
st.markdown(
    "This tool uses **Season-to-Date Averages** scraped directly from Basketball Reference to calculate player value via the Composite Z-Score method.")
st.divider()

# Calculate and display rankings (using the imported function)
fantasy_rankings, z_cols = calculate_z_scores(df_raw, FANTASY_STATS)

if fantasy_rankings.empty:
    st.warning("Cannot calculate rankings. Please check data scraping status above.")
else:

    # --- Main Section: Roster Selection Analysis (Z-Score) ---
    st.header("1. Composite Z-Score Ranking")
    st.markdown("""
    The **Composite Z-Score** standardizes player performance across all selected categories (PTS, REB, AST, STL, BLK, TOV). 
    A higher score indicates a better overall fantasy player relative to the field. 
    *(Note: Turnovers are inverted, so a low TOV results in a positive Z-Score contribution)*.
    """)

    # Display the current data used for analysis
    st.subheader("Current Season-to-Date Averages Used")
    st.dataframe(df_raw.set_index('player_name')[FANTASY_STATS].head(10), use_container_width=True)

    # Display the final ranking table
    st.subheader("Full Fantasy Ranking Table")
    display_cols = ['rank_Zscore'] + z_cols + ['composite_Zscore']
    st.dataframe(
        fantasy_rankings[display_cols].style.background_gradient(cmap='RdYlGn', subset=['composite_Zscore']),
        use_container_width=True,
        column_config={
            "Rank": st.column_config.NumberColumn(
                "Rank",
                format="%d",
            ),
            "Composite_Z_Score": st.column_config.NumberColumn(
                "Composite Z-Score",
                format="%.2f",
            ),
            "PTS_Z": "Points Z-Score",
            "TRB_Z": "Rebounds Z-Score",
            "AST_Z": "Assists Z-Score",
            "STL_Z": "Steals Z-Score",
            "BLK_Z": "Blocks Z-Score",
            "TOV_Z": "TOV (Neg) Z-Score"
        }
    )

    st.success(
        f"**Optimal Roster Pick:** The highest ranked player based on current season data is **{fantasy_rankings.index[0]}**.")

    # --- Sidebar Controls for Comparison Analysis ---
    st.sidebar.header("Player Comparison Controls")

    top_players = fantasy_rankings.index.tolist()

    player_comparison_selection = st.sidebar.multiselect(
        "Select Players to Compare",
        options=top_players,
        default=top_players[:4]
    )

    # --- Main Section: Player Comparison ---
    st.header("2. Interactive Z-Score Comparison")
    st.markdown("Select players in the sidebar to visualize how their Composite Z-Scores compare against each other.")

    comparison_fig = plot_z_score_comparison(fantasy_rankings, player_comparison_selection)

    if comparison_fig:
        st.pyplot(comparison_fig)
    else:
        st.info("Select players in the sidebar to generate a comparison chart.")

st.markdown("---")
st.caption("Application built for NBA data analysts using Streamlit and live-scraped data.")
