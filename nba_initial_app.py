# Import the data processing functions from the separate file
import pandas as pd
from data_processing import join_dataframes, calculate_z_scores, merge_dataframes
from plots_for_app import plot_z_score_comparison
from roster_optimizer import *
import streamlit as st

# Load required dataframes #

# Load the newest data in the background
df_hist = pd.read_csv('data/stats_hist.csv')
df_raw = join_dataframes()
TEAM_COL = ['team']
FGP_COL = ['fg_pts']
COST_COL = ['current_cost']
FANTASY_STATS = ['pts', 'trb', 'ast', 'stl', 'blk']
SEL_COLS = TEAM_COL + FGP_COL + COST_COL + FANTASY_STATS

# --- 1. Streamlit Application Layout --- #

# Configuration and Initialization
st.set_page_config(layout="wide", page_title="NBA Fantasy Analysis App")
st.title("🏀 NBA Fantasy Analysis 🏀")
st.markdown(
    "This tool uses **Season-to-Date Averages** scraped directly from Basketball Reference Page to calculate player value.")
# st.divider()
tab1, tab2, tab3 = st.tabs(["📊 Statistics", "⚛ Roster Simulator", "📈 Player Analysis"])

# st.session_state.historical_data = []
st.session_state.report = pd.DataFrame()

# --- Tab 1: Season Statistics ---
with tab1:
    tab1.header("1. Data Loading")
    col1, col2, col3 = tab1.columns([3, 3, 2], vertical_alignment="center")

    # Initialize Session State for History (ensures data persists through reruns)
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = df_hist

    with col1:
        st.subheader("Last Historical Records")
        st.dataframe(
            st.session_state.historical_data[['player_name', 'pts', 'etl_date']].sort_values(by = ['etl_date', 'pts'], ascending= [False, False]).head(5),
            use_container_width=True
        )

    with col2:
        st.subheader("Latest Data - Sample")
        st.dataframe(
            df_raw[['player_name', 'pts', 'etl_date']].sort_values(by = ['etl_date', 'pts'], ascending= [False, False]).head(5),
            use_container_width=True
        )

    with col3:
        st.metric(
            label="Historical Data Length",
            value=f"{len(st.session_state.historical_data)} rows",
            delta=f"+{len(df_raw)} rows pending append"
        )

        # row1, row2, row3 = st.rows((1, 2, 3))
        st.markdown(
            "Use the button below to append new data to the historical.")

        # Button logic: When clicked, run load_hist and update session state
        if st.button("Append New Data to History", type="primary"):
            # The history data is pulled from session state
            history_data = st.session_state.historical_data

            # Call the new function (combining data and adding date)
            combined_data = merge_dataframes(history_data, df_raw)

            # Update session state with the combined data
            st.session_state.historical_data = combined_data
            st.success(f"Data successfully appended! New total history length: {len(combined_data)} rows.")

    tab1.subheader("NBA 2025/2026 Season - Sample Data")
    df_hist = st.session_state.historical_data

    # Display the current data used for analysis
    if df_hist.empty:
        st.warning("No data to display")
    else:
        tab1.dataframe(df_hist.set_index('player_name')[SEL_COLS].head(5), width='stretch')

    # Calculate and display rankings (using the imported function)
    fantasy_rankings, z_cols = calculate_z_scores(df_raw, FANTASY_STATS, COST_COL, FGP_COL)

    if fantasy_rankings.empty:
        st.warning("Cannot calculate rankings. Please check data scraping status above.")
    else:
        tab1.header("2. Composite Score Ranking")
        tab1.markdown("""
        The **Composite Z-Score** standardizes player performance across all selected categories (PTS, REB, AST, STL, BLK). 
        A higher score indicates a better overall fantasy player relative to the field. 
        """)

    # Display the final ranking table
    tab1.subheader("Full Fantasy Ranking Table")
    display_cols = TEAM_COL + COST_COL + FGP_COL + ['pts_per_cost'] + ['composite_Zscore'] + z_cols + ['cost_Zscore']
    tab1.dataframe(
        fantasy_rankings[display_cols].style.background_gradient(cmap='RdYlGn', subset=['pts_per_cost','composite_Zscore']),
        use_container_width=True,
        column_config={
            "team": "Team",
            "current_cost": st.column_config.NumberColumn(
                "Cost",
                format="%.2f",
            ),
            "fg_pts": st.column_config.NumberColumn(
                "FG Pts",
                format="%.2f",
            ),
            "pts_per_cost": st.column_config.NumberColumn(
                "Pts per Cost (PPC)",
                format="%.2f",
            ),
            "composite_Zscore": st.column_config.NumberColumn(
                "Composite Score",
                format="%.2f",
            ),
            "pts_Z": "Points Score",
            "trb_Z": "Rebounds Score",
            "ast_Z": "Assists Score",
            "stl_Z": "Steals Score",
            "blk_Z": "Blocks Score",
            "cost_Zscore": "Cost Score",
        }
    )

# --- Tab 2: Roaster Simulation---
with tab2:
    tab2.subheader("2. Roster Simulation")

    max_games = df_raw['games_played'].max()
    budget = 100

    col1, col2, col3 = st.columns(3)
    with col1:
        week_start = st.number_input(
            "Week Start", min_value=1, max_value=25, value=1, step=1, placeholder = "Type a week..."
        )

    with col2:
        num_weeks = st.number_input(
            "# Weeks", value=1, placeholder="Type length of projection..."
        )

    starting_r = df_raw.loc[df_raw['in_current_roster'] == 1, 'player_name'].tolist()
    week_len = week_start + num_weeks - 1
    week_columns = [
        f'{i}_{j}'
        for i in range(week_start, week_len + 1)
        for j in range(1, 8)
    ]

    # starting = ['Nikola Jokic', 'Victor Wembanyama', 'Shai Gilgeous-Alexander', 'Ryan Rollins', 'Kon Knueppel',
    #             'Jeremiah Fears', 'Ryan Kalkbrenner', 'Collin Gillespie', 'Neemias Queta', 'Mouhamed Gueye']

    df_filter = pre_select_options(df=fantasy_rankings,
                                   n_weeks=num_weeks,
                                   week_start=week_start,
                                   starting_roster=starting_r,
                                   days = week_columns)

    with col3:
        # Button logic: When clicked, run roster optimization and update session state
        if st.button("Run Roster Optimization", type="primary"):
            st.info("--- Starting Optimization ---")

            # Call the optimization function
            optimized_roster_df = optimize_roster_multiweek(
                budget=budget,
                starting_roster=starting_r,
                df=df_filter,
                start_week=week_start,
                num_weeks=num_weeks,
                days=week_columns,
                obj_var='fg_pts',
                max_swaps=2,
                verbose=True
            )
            report = get_detailed_report(optimized_roster_df, df_filter)

            # st.session_state.optimized_roster = optimized_roster_df
            st.session_state.result_report = report
            st.info("--- Optimization Finished ---")

        df_report = st.session_state.result_report
        df_roster = st.session_state.optimized_roster

    if not df_report.empty:
        st.subheader("Optimized Roster")
        st.dataframe(df_report,use_container_width=True)

        # if not df_roster.empty:
        #     # Verify daily constraints on the optimal roster
        #     for day in week_columns:
        #         bc_count = df_roster[
        #             (df_roster['position'] == 'Backcourt') & (df_roster[day] == 1)].shape[0]
        #         fc_count = df_roster[
        #             (df_roster['position'] == 'Frontcourt') & (df_roster[day] == 1)].shape[0]
        #         total_count = bc_count + fc_count
        #         status = "OK (5 total, 3/2 or 2/3 split)" if total_count == 5 and (
        #                     (bc_count == 3 and fc_count == 2) or (
        #                     bc_count == 2 and fc_count == 3)) else "N/A (Total playing != 5)" if total_count != 0 else "N/A (No players playing)"
        #             # print(f"{day}: BC={bc_count}, FC={fc_count}, Total={total_count} -> {status}")
        #     st.success(f"Roster Optimization Status: {status}")

        # if df_roster.empty:
        #     st.info("--- Roster did not run ---")
        # else:
        #     st.subheader("Optimized Roster")
        #     # Roster cost
        #     check_cost = df_roster['current_cost'].sum()
        #     st.markdown(f"**Current Cost:** {check_cost}")
        #
        #     st.dataframe(
        #         df_roster,
        #         use_container_width=True
        #     )

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
