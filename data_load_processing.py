import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO


# @st.cache_data
def scrape_current_data():
    """
    Scrapes the NBA 'Per Game' statistics for the 2025 season from Basketball Reference.
    The function uses st.info/st.error for status updates in the Streamlit environment.
    """
    url = 'https://www.basketball-reference.com/leagues/NBA_2025_per_game.html'
    st.info(f"Data scraped from: {url}")

    try:
        # pd.read_html automatically finds tables on the webpage
        data = pd.read_html(url)
    except Exception as e:
        st.error(f"Error scraping data from Basketball Reference. Check URL or internet connection. Error: {e}")
        # Fallback to an empty DataFrame if scraping fails
        return pd.DataFrame({
            'Player': [], 'PTS': [], 'TRB': [], 'AST': [], 'STL': [], 'BLK': [], 'TOV': []
        })

    # The main player stats table is usually the first one found
    df = data[0]

    # Data Cleaning and Filtering
    # 1. Filter out repeated header rows within the table (where 'Player' equals 'Player')
    df = df[df['Player'] != 'Player'].copy()
    df = df[~df['Player'].str.contains('Average')].copy()

    # 2. Rename columns for easier maintenance and normalize players names
    df = df.rename(
        columns={'Rk': 'week_rank', 'Player': 'player_name', 'Age': 'player_age', 'Team': 'team',
                 'Pos': 'position','G': 'games_played', 'GS': 'games_started', 'MP': 'minutes_played',
                 'FG': 'fg', 'FGA': 'fga', 'FG%': 'fg_pcnt', '3P': '3p', '3PA': '3pa', '3P%': '3p_pcnt',
                 '2P': '2p', '2PA': '2pa', '2P%': '2p_pcnt', 'eFG%': 'efg_pnt', 'FT': 'ft', 'FTA': 'fta',
                 'FT%': 'ft_pcnt', 'ORB': 'orb', 'DRB': 'drb', 'TRB': 'trb', 'AST': 'ast', 'STL': 'stl',
                 'BLK': 'blk', 'TOV': 'tov', 'PF': 'pf', 'PTS': 'pts', 'Awards': 'awards'}
        )

    # Leave space to handle players names and unusual characters #

    # 3. Create main stat column (fantasy points)
    df['pts_fg'] = df['pts'] + df['trb'] + 2*df['ast'] + 3*df['blk'] + 3*df['stl']

    # 3. Select relevant columns for 9-Category H2H fantasy scoring
    cols_to_keep = ['player_name', 'pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'games_played', 'minutes_played', 'games_started',
                    'fg', 'fga', 'fg_pcnt', '3p', '3pa', '3p_pcnt', '2p', '2pa', '2p_pcnt', 'efg_pnt','pts_fg']

    # Ensure the required columns exist before selecting
    missing_cols = [col for col in cols_to_keep if col not in df.columns]
    if missing_cols:
        st.warning(f"The following columns were missing after scraping: {missing_cols}. Using available data.")
        # Attempt to proceed with available columns, but for this specific source, they should exist.
        cols_to_keep = [col for col in cols_to_keep if col in df.columns]

    df = df[cols_to_keep]

    # 3. Convert statistical columns to numeric
    stats_cols = [col for col in cols_to_keep if col != 'player_name']
    for col in stats_cols:
        # Coerce errors to NaN for players who might have null values
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Drop rows where key stats are NaN (e.g., players with very little play time)
    df = df.dropna(subset=stats_cols)

    st.success(f"Successfully scraped data for {len(df)} players.")
    return df

# @st.cache_data
def calculate_z_scores(df_data, stats_cols):
    """
    Calculates the Z-Score for each player across key fantasy categories
    based on the season-to-date per-game averages.
    """
    if df_data.empty:
        return pd.DataFrame(), []

    df_analysis = df_data.copy().set_index('player_name')

    # Filter only players who have played enough games/minutes to be considered
    # A simple filter: only include players with a non-zero TRB, AST, etc.
    df_analysis = df_analysis[(df_analysis['pts'] > 0) & (df_analysis['pts_fg'] > 0)]

    for stat in stats_cols:
        # Calculate mean and standard deviation across all filtered players
        mean_val = df_analysis[stat].mean()
        std_val = df_analysis[stat].std()

        # Calculate Z-Score
        z_score_col = f'{stat}_Z'
        if std_val == 0:
            df_analysis[z_score_col] = 0
        else:
            df_analysis[z_score_col] = (df_analysis[stat] - mean_val) / std_val

        # For Turnovers (TOV), lower is better, so invert the Z-score sign
        if stat == 'tov':
            df_analysis[z_score_col] *= -1

    # Calculate Composite Z-Score (Sum Z-scores across all categories)
    z_cols = [col for col in df_analysis.columns if '_Z' in col]
    df_analysis['composite_Zscore'] = df_analysis[z_cols].sum(axis=1)

    # Rank the players
    df_analysis = df_analysis.sort_values(by='composite_Zscore', ascending=False)
    df_analysis['rank_Zscore'] = np.arange(1, len(df_analysis) + 1)

    return df_analysis, z_cols
