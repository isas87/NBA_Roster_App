# Import the data processing functions from the separate file
from data_processing import join_dataframes, calculate_z_scores
from plots_for_app import plot_z_score_comparison
from roster_optimizer import optimize_nba_roster
import streamlit as st

# Call the imported data loading function
df_raw = join_dataframes()

MAIN_COLS = ['team','fg_pts']
COST_COL = ['current_cost']
FANTASY_STATS = ['pts', 'trb', 'ast', 'stl', 'blk']
SEL_COLS = MAIN_COLS+COST_COL+FANTASY_STATS

# Calculate and display rankings (using the imported function)
fantasy_rankings, z_cols = calculate_z_scores(df_raw, FANTASY_STATS, COST_COL)

# --- 1. Streamlit Application Layout --- #

# Configuration and Initialization
st.set_page_config(layout="wide", page_title="NBA Fantasy Roster")
st.title("🏀 NBA Fantasy Roster Analyzer")
st.markdown(
    "This tool uses **Season-to-Date Averages** scraped directly from Basketball Reference Page to calculate player value via the Composite Z-Score method.")
st.divider()

tab1, tab2, tab3 = st.tabs(["🗃 Statistics", "Roster Simulator", "Plot Stats"])

# --- Tab 1: Season Statistics ---
with tab1:
    tab1.subheader("NBA 2025/206 Season - Sample Data")

    # Display the current data used for analysis
    if df_raw.empty:
        st.warning("No data to display")
    else:
        tab1.dataframe(df_raw.set_index('player_name')[SEL_COLS].head(10), width='stretch')

    if fantasy_rankings.empty:
        st.warning("Cannot calculate rankings. Please check data scraping status above.")
    else:
        tab1.subheader("1. Composite Z-Score Ranking")
        tab1.markdown("""
        The **Composite Z-Score** standardizes player performance across all selected categories (PTS, REB, AST, STL, BLK). 
        A higher score indicates a better overall fantasy player relative to the field. 
        """)

    # Display the final ranking table
    tab1.subheader("Full Fantasy Ranking Table")
    display_cols = MAIN_COLS + COST_COL+['rank_cost']+['pts_per_cost'] + ['rank_score'] + ['composite_Zscore'] + z_cols+['cost_Zscore']
    tab1.dataframe(
        fantasy_rankings[display_cols].style.background_gradient(cmap='RdYlGn', subset=['pts_per_cost','composite_Zscore']),
        use_container_width=True,
        column_config={
            "fg_pts": st.column_config.NumberColumn(
                "FG Points",
                format="%.2f",
            ),
            "current_cost": st.column_config.NumberColumn(
                "Cost",
                format="%.2f",
            ),
            "pts_per_cost": st.column_config.NumberColumn(
                "Points per Cost",
                format="%.2f",
            ),
            "rank_cost": st.column_config.NumberColumn(
                "Rank Cost",
                format="%d",
            ),
            "rank_score": st.column_config.NumberColumn(
                "Rank Score",
                format="%d",
            ),
            "composite_Zscore": st.column_config.NumberColumn(
                "Composite Z-Score",
                format="%.2f",
            ),
            "pts_Z": "Points Z-Score",
            "trb_Z": "Rebounds Z-Score",
            "ast_Z": "Assists Z-Score",
            "stl_Z": "Steals Z-Score",
            "blk_Z": "Blocks Z-Score",
            "cost_Zscore": "Cost Z-Score",
        }
    )

    # st.success(
    #     f"**Optimal Roster Pick:** The highest ranked player based on current season data is **{fantasy_rankings.index[0]}**.")

# --- Tab 2: Roaster Simulation---
with tab2:
    tab2.subheader("2. Roster Simulation")

# --- Tab 3: Plot stats---
with tab3:

    tab3.markdown(
        "Select players in the sidebar to visualize how their Composite Z-Scores compare against each other."
    )

    top_players = fantasy_rankings.index.tolist()

    player_comparison_selection = tab3.multiselect(
        "Select players in the sidebar",
        top_players,
    )

    #
    # player_comparison_selection = st.sidebar.multiselect(
    #     "Select Players to Compare",
    #     options=top_players,
    #     default=top_players[:4]
    # )

    comparison_fig = plot_z_score_comparison(fantasy_rankings, player_comparison_selection)

    if comparison_fig:
        st.pyplot(comparison_fig)
    else:
        st.info("Select players in the sidebar to generate a comparison chart.")

st.markdown("---")
st.caption("Application built for NBA data analysts using Streamlit and live-scraped data.")
