from data_loading import *
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def join_dataframes():
    df_stats= scrape_current_data()
    df_cost = load_cost_data()
    df_sched = load_schedule_data()
    df_data = pd.merge(df_stats, df_cost, on='player_name')
    df_data = pd.merge(df_data, df_sched, how = 'left', left_on = 'team', right_on = 'week_day')

    if df_data.empty:
        st.warning("Error on joining datasets")

    df_data = df_data.sort_values(by='fg_pts', ascending=False)
    return df_data

@st.cache_data
def calculate_z_scores(df_data, stats_cols, cost_col):
    """
    Calculates the Z-Score for each player across key fantasy categories
    based on the season-to-date per-game averages.
    """
    pd.options.display.float_format = "{:,.2f}".format

    if df_data.empty:
        return pd.DataFrame(), []

    df_analysis = df_data.copy().set_index('player_name')

    # Filter only players who have played enough games/minutes to be considered
    df_analysis = df_analysis[(df_analysis['pts'] > 0) & (df_analysis['fg_pts'] > 0)]

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

        # # For Turnovers (TOV), lower is better, so invert the Z-score sign
        # if stat == 'tov':
        #     df_analysis[z_score_col] *= -1

    # Calculate Composite Z-Score (Sum Z-scores across all categories)
    z_cols = [col for col in df_analysis.columns if '_Z' in col]
    # df_analysis['composite_Zscore'] = df_analysis[z_cols].sum(axis=1)
    df_analysis['composite_Zscore'] = df_analysis['pts_Z'] + df_analysis['trb_Z'] + 2*df_analysis['ast_Z'] + 3*df_analysis['blk_Z'] + 3*df_analysis['stl_Z']

    df_analysis['cost_Zscore'] = (df_analysis[cost_col].max() - df_analysis[cost_col]) / (df_analysis[
        cost_col].max() - df_analysis[cost_col].min())+1
    df_analysis['pts_per_cost'] = df_analysis['composite_Zscore'] / df_analysis['cost_Zscore']

    # Rank the players by Cost and by Composite Score
    df_analysis = df_analysis.sort_values(by='pts_per_cost', ascending=False)
    df_analysis['rank_pts_cost'] = np.arange(1, len(df_analysis) + 1)
    df_analysis = df_analysis.sort_values(by='composite_Zscore', ascending=False)
    df_analysis['rank_score'] = np.arange(1, len(df_analysis) + 1)
    df_analysis = df_analysis.sort_values(by='current_cost', ascending=False)
    df_analysis['rank_cost'] = np.arange(1, len(df_analysis) + 1)

    return df_analysis, z_cols
