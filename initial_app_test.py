import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Data Loading and Preparation ---

@st.cache_data
def load_mock_weekly_data():
    """
    Simulates loading weekly aggregated player performance data.
    
    NOTE: In a real-world scenario, you would replace this with a function 
    to scrape the current weekly data (e.g., using pandas.read_html 
    or a dedicated NBA API wrapper) and merge it with your historical dataset.
    """
    data = {
        'Player': ['Jokic', 'Jokic', 'Jokic', 'Jokic', 'LeBron', 'LeBron', 'LeBron', 'LeBron', 'Hali', 'Hali', 'Hali', 'Hali', 'Gobert', 'Gobert', 'Gobert', 'Gobert'],
        'Week': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
        'PTS': [28.5, 26.1, 30.2, 29.5, 24.3, 27.5, 22.0, 25.0, 18.0, 16.5, 20.1, 19.8, 12.0, 15.1, 10.5, 13.0],
        'TRB': [13.2, 12.8, 14.0, 13.5, 7.5, 8.0, 6.8, 7.0, 4.0, 3.5, 4.2, 3.9, 14.5, 13.9, 15.0, 14.2],
        'AST': [9.0, 9.5, 10.1, 11.0, 8.5, 7.9, 9.0, 8.5, 13.0, 14.5, 11.5, 12.0, 1.5, 1.2, 1.0, 1.1],
        'STL': [1.5, 1.8, 1.0, 1.2, 1.0, 0.9, 1.1, 1.0, 1.5, 2.0, 1.8, 1.9, 0.8, 0.7, 0.9, 0.6],
        'BLK': [0.8, 0.6, 1.0, 0.9, 0.5, 0.6, 0.4, 0.5, 0.2, 0.3, 0.1, 0.2, 2.5, 2.2, 2.8, 2.4],
        'TOV': [3.5, 4.0, 3.1, 3.2, 3.0, 3.1, 2.5, 2.8, 2.5, 2.0, 2.8, 2.6, 1.0, 0.9, 1.1, 1.0], # Turnovers (lower is better)
    }
    df = pd.DataFrame(data)
    df = df.sort_values(by=['Player', 'Week']).reset_index(drop=True)
    return df

@st.cache_data
def calculate_z_scores(df_data, stats_cols):
    """
    Calculates the Z-Score for each player across key fantasy categories 
    based on the latest available week's data.
    """
    latest_week = df_data['Week'].max()
    df_analysis = df_data[df_data['Week'] == latest_week].copy().set_index('Player')
    
    st.subheader(f"Data from Latest Week: {latest_week}")
    st.dataframe(df_analysis[stats_cols], use_container_width=True)

    for stat in stats_cols:
        # Calculate mean and standard deviation across all players for that stat
        mean_val = df_analysis[stat].mean()
        std_val = df_analysis[stat].std()
        
        # Calculate Z-Score
        z_score_col = f'{stat}_Z'
        # Handle division by zero if std_val is 0 (unlikely with enough players, but good practice)
        if std_val == 0:
             df_analysis[z_score_col] = 0
        else:
             df_analysis[z_score_col] = (df_analysis[stat] - mean_val) / std_val
        
        # For Turnovers (TOV), lower is better, so invert the Z-score sign
        if stat == 'TOV':
            df_analysis[z_score_col] *= -1

    # Calculate Composite Z-Score (Sum Z-scores across all categories)
    z_cols = [col for col in df_analysis.columns if '_Z' in col]
    df_analysis['Composite_Z_Score'] = df_analysis[z_cols].sum(axis=1)

    # Rank the players
    df_analysis = df_analysis.sort_values(by='Composite_Z_Score', ascending=False)
    df_analysis['Rank'] = np.arange(1, len(df_analysis) + 1)
    
    return df_analysis, z_cols

# --- 2. Trend Analysis Plotting Function ---

def plot_player_trend(df, player_name, stat, window):
    """
    Calculates the rolling average and generates a Matplotlib figure.
    """
    # Filter data for the specific player
    player_df = df[df['Player'] == player_name].copy()
    
    # Calculate the rolling average (the trend)
    player_df[f'{stat}_Rolling_Avg'] = player_df[stat].rolling(window=window, min_periods=1).mean()
    
    # Plotting the trend
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot weekly performance
    ax.plot(player_df['Week'], player_df[stat], 
             marker='o', linestyle='--', color='gray', 
             label=f'{stat} (Weekly)')
    
    # Plot rolling average trend
    ax.plot(player_df['Week'], player_df[f'{stat}_Rolling_Avg'], 
             marker='D', linestyle='-', color='#DB342A', linewidth=3,
             label=f'{stat} ({window}-Week Avg)')
    
    ax.set_title(f'{player_name} Performance Trend: {stat} Over Time', fontsize=16)
    ax.set_xlabel('Week of Season', fontsize=12)
    ax.set_ylabel(f'{stat} Per Week', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_xticks(player_df['Week'])
    plt.tight_layout()
    
    return fig


# --- 3. Streamlit Application Layout ---

# Configuration and Initialization
st.set_page_config(layout="wide", page_title="NBA Fantasy Analyzer")
df_raw = load_mock_weekly_data()
FANTASY_STATS = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV']
all_players = df_raw['Player'].unique().tolist()
window_min = 2
window_max = len(df_raw['Week'].unique())

st.title("🏀 NBA Fantasy Roster Analyzer")
st.markdown("Use the controls on the sidebar for interactive trend analysis and view the Composite Z-Score ranking below for team selection.")
st.divider()

# --- Sidebar Controls for Trend Analysis ---
st.sidebar.header("Trend Analysis Controls")

player_selection = st.sidebar.selectbox(
    "Select Player",
    all_players,
    index=0 # Default to Jokic
)

stat_selection = st.sidebar.selectbox(
    "Select Statistic",
    FANTASY_STATS,
    index=0 # Default to Points
)

rolling_window = st.sidebar.slider(
    "Rolling Average Window (Weeks)",
    min_value=window_min,
    max_value=window_max,
    value=3,
    step=1
)
st.sidebar.caption("The rolling average helps identify recent hot/cold streaks.")

# --- Main Section: Trend Analysis ---
st.header("1. Player Performance Trends")
st.info(f"Analyzing {player_selection}'s trend in **{stat_selection}** using a **{rolling_window}**-week rolling average.")

# Generate and display the plot
if player_selection in all_players:
    trend_fig = plot_player_trend(df_raw, player_selection, stat_selection, rolling_window)
    st.pyplot(trend_fig)
else:
    st.warning("Please select a player to view the trend analysis.")


# --- Main Section: Roster Selection Analysis (Z-Score) ---
st.header("2. Composite Z-Score Ranking")
st.markdown("""
The **Composite Z-Score** standardizes player performance across all selected categories (PTS, REB, AST, STL, BLK, TOV). 
A higher score indicates a better overall fantasy player relative to the rest of the players in the dataset. 
*(Note: Turnovers are inverted, so a low TOV results in a positive Z-Score contribution)*.
""")

# Calculate and display rankings
fantasy_rankings, z_cols = calculate_z_scores(df_raw, FANTASY_STATS)

# Display the final ranking table
display_cols = ['Rank'] + z_cols + ['Composite_Z_Score']
st.dataframe(
    fantasy_rankings[display_cols].style.background_gradient(cmap='RdYlGn', subset=['Composite_Z_Score']),
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

st.success(f"**Optimal Roster Pick:** The highest ranked player based on the current data is **{fantasy_rankings.index[0]}**.")
st.markdown("---")
st.caption("Application built for NBA data analysts using Streamlit and Python.")
