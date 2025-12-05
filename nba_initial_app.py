# Import the data processing functions from the separate file
import pandas as pd
import plotly.express as px
from datetime import datetime
import time

from IPython.utils import wildcard

from data_processing import *
from roster_optimizer import *
# from trend_performance_plot import *
from trend_analysis import *
import streamlit as st

# Initialize session state for chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None

# Load the newest data in the background
df_hist = pd.read_csv('data/stats_hist.csv', index_col = 0 )
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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistics", "⚛ Roster Simulator", "📈 Player Analysis", "🤖 AI Assistant"])

# st.session_state.historical_data = []
st.session_state.result_report = pd.DataFrame()
st.session_state.optimized_roster = pd.DataFrame()

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
            hide_index=True
        )

    with col2:
        st.subheader("Latest Data - Sample")
        st.dataframe(
            df_raw[['player_name', 'pts', 'etl_date']].sort_values(by = ['etl_date', 'pts'], ascending= [False, False]).head(5),
            hide_index=True
        )

    with col3:
        st.metric(
            label="Historical Data Length",
            value=f"{len(st.session_state.historical_data)} rows",
            delta=f"+{len(df_raw)} rows pending append"
        )

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
    display_cols = ['player_name'] + TEAM_COL + ['position'] + COST_COL + FGP_COL + ['pts_per_cost'] + ['composite_Zscore'] + z_cols + ['cost_Zscore']
    tab1.dataframe(
        fantasy_rankings[display_cols].style.background_gradient(cmap='RdYlGn', subset=['pts_per_cost','composite_Zscore']),
        hide_index=True,
        column_config={
            'player_name': 'Player',
            "team": "Team",
            'position': 'Position',
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

    col1, col2, col3, col4, col5 = st.columns(5)
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

    with col3:
        min_pts_ = st.number_input(
            "Min Rank FGP",
            value=200, placeholder="Min ranking based on FG points"
        )

    with col4:
        min_score_ = st.number_input(
            "Min Rank Score",
            value=200, placeholder="Min ranking based on Score"
        )

    with col5:
        st.markdown("Check this box to run the Wildcard Model")

        wc_ = st.checkbox("Wildcard Option", value=False,
                    # key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False,
                    label_visibility="visible", width="content")

        st.markdown(f"Wildcard selected {wc_}")

    df_filter = pre_select_options(df=fantasy_rankings,
                                   n_weeks=num_weeks,
                                   min_rank_pts=min_pts_,
                                   min_rank_scr=min_score_,
                                   starting_roster=starting_r,
                                   days = week_columns,
                                   wildcard=wc_)

    # Button logic: When clicked, run roster optimization and update session state
    if st.button("Run Roster Optimization", type="primary"):
        start = datetime.now()

        st.info("--- Starting Optimization ---", icon="ℹ️"
                # "Start time:", start.strftime("%Y-%m-%d %H:%M")
                )
        st.write("Start time:", start.strftime("%Y-%m-%d %H:%M"))

        # Call the optimization function
        optimized_roster_df = optimize_roster_multiweek(
            budget=budget,
            starting_roster=starting_r,
            df=df_filter,
            start_week=week_start,
            num_weeks=num_weeks,
            days=week_columns,
            obj_var='fg_pts',
            wildcard=wc_,
            max_swaps=2,
            verbose=True
            )

        lineup,changes  = get_detailed_report(optimized_roster_df, df_filter)

        st.session_state.optimized_roster = lineup
        st.session_state.result_report = changes
        st.info("--- Optimization Finished ---")

    df_roster = st.session_state.optimized_roster
    df_report = st.session_state.result_report

    if not df_report.empty:

        col5, col6= st.columns([1,3])

        with col5:
            st.subheader("Optimized Roster - Summary")
            st.write("Week:", changes['week'][0])
            if wildcard:
                st.write("Roster", changes['full_lineup'][0] )
            else:
                st.write("No Swaps:", changes['num_swaps'][0])
                st.write("Removed Players: -->", changes['players_removed'][0])
                st.write("Added Players: <--", changes['players_added'][0])
                st.write("Details:", changes['swap_details'][0])
            st.write("Week Points:", changes['week_points'][0])
            st.write("Total Cost", changes['final_cost'][1])

        with col6:
            st.subheader("Optimized Roster Lineup")
            st.dataframe(df_roster,
                         width='stretch',
                         column_config = {
                            'week': 'Week',
                            'day': 'Day',
                            'lineup_config': 'Line up',
                            'players_active': 'Active Players',
                            'backcourt_active': 'Backcourt Active',
                            'frontcourt_active': 'Frontcourt Active',
                            'points':st.column_config.NumberColumn('FG Points', format="%.2f")
                }
            )

# --- Tab 3: Plot stats---
with tab3:
    st.subheader("Performance Over Time")

    tab3.markdown(
        "Select players to visualize how their Composite Z-Scores compare against each other."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
    # Player selection
        all_players = sorted(df_hist['player_name'].unique())
        selected_players = st.multiselect(
            "Select Players",
            all_players,
            default=all_players[:3] if len(all_players) >= 3 else all_players
        )

    with col2:
        # Date range
        min_date = pd.to_datetime(df_hist['etl_date']).min()
        max_date = pd.to_datetime(df_hist['etl_date']).max()

        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

    with col3:
        metric = st.selectbox(
            "Select Metric",
            ['points', 'assists', 'rebounds', 'blocks', 'steals' , 'minutes played', 'fantasy points']
        )

    # Filter data
    if len(date_range) == 2:
        mask = (
                (df_hist['player_name'].isin(selected_players)) &
                (pd.to_datetime(df_hist['etl_date']) >= pd.to_datetime(date_range[0])) &
                (pd.to_datetime(df_hist['etl_date']) <= pd.to_datetime(date_range[1]))
        )
        filtered_df = df_hist[mask]
    else:
        df_hist['etl_date'] = pd.to_datetime(df_hist['etl_date'])
        filtered_df = df_hist[df_hist['player_name'].isin(selected_players)].sort_values(by = ['etl_date'])

    if filtered_df.empty:
        st.warning("No data available for selected filters")
    else:
        c1, c2 = st.columns(2)
    if metric == 'points':
        metric_name = 'pts'
    elif metric == 'assists':
        metric_name = 'ast'
    elif metric == 'rebounds':
        metric_name = 'trb'
    elif metric == 'blocks':
        metric_name = 'blk'
    elif metric == 'steals':
        metric_name = 'stl'
    elif metric == 'minutes played':
        metric_name = 'minutes_played'
    elif metric == 'fantasy points':
        metric_name = 'fg_pts'

    with c1:
        st.subheader("📈Performance Over Time")
        # Line chart
        fig=px.line(
            filtered_df,
            x='etl_date',
            y=metric_name,
            color='player_name',
            markers=True,
            title=f'Avg. {metric.title()} Over Time',
            labels={'etl_date': 'Date', metric_name: metric.title()}
        )
        st.plotly_chart(fig)

    with c2:
        st.subheader("📊 Consistency Comparison")

        df_c = calculate_consistency(filtered_df, selected_players, metric_name)

        fig_bar = px.bar(
            df_c,
            x='player_name',
            y=metric_name+'_consistency',
            color='player_name',
            title=f'Avg. {metric.title()} Consistency',
            labels={'player_name': 'Player', metric_name: metric.title()}
        )
        st.plotly_chart(fig_bar)

    st.subheader("📈Stats Summary")

    df_c = calculate_trend(filtered_df, selected_players, metric_name)

    st.dataframe(df_c,
                 width='content',
                 hide_index=True,
                 column_config={
                     'player_name': 'Player',
                     'trend': 'Trend',
                     'r2_value': 'R2',
                     'p_value': 'p-Value',
                     'slope': 'Slope Trend',
                     'best': f'Best {metric.title()}',
                     'worst': f'Worst {metric.title()}'
                 }
            )

# --- Tab 4: Chatbot---
with tab4:
    st.header("🤖 AI Performance Assistant")

    st.markdown("""
                Ask me anything about player performance! I can help with:
                - Player statistics and comparisons
                - Performance trends and improvements
                - Recommendations for players to watch
                - Head-to-head matchup analysis
                """)

    # Chat interface
    st.subheader("Chat with AI Assistant")

    # Display chat history
    chat_container = st.container()

    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message['role'] == 'user':
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**AI Assistant:** {message['content']}")
            st.markdown("---")

    # Chat input
    user_query = st.chat_input("Ask me about player performance...")

    if user_query:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_query,
            'timestamp': datetime.now()
        })

st.markdown("---")
st.caption("Application built for NBA data analysts using Streamlit and live-scraped data.")
