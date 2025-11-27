import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


# Compare players at specific time points
def compare_players_snapshot(df, players, date=None):
    """Compare multiple players at a specific snapshot"""
    if date is None:
        date = df['etl_date'].max()

    comparison_df = df[
        (df['player_name'].isin(players)) &
        (df['etl_date'] == date)
        ][['player_name', 'fg_pts','pts', 'ast', 'trb', 'blk', 'stl']]

    return comparison_df

def plot_z_score_comparison(df_rankings, players):
    """
    Generates a Matplotlib bar chart comparing the Composite Z-Scores of selected players.
    """
    if df_rankings.empty or not players:
        return None

    df_plot = df_rankings.loc[df_rankings.index.intersection(players), ['composite_Zscore']].sort_values(
        'composite_Zscore', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['#DB342A' if score == df_plot['composite_Zscore'].max() else '#3498DB' for score in
              df_plot['composite_Zscore']]

    ax.bar(df_plot.index, df_plot['composite_Zscore'], color=colors)

    ax.set_title('Composite Z-Score Comparison', fontsize=16)
    ax.set_xlabel('Player', fontsize=12)
    ax.set_ylabel('Composite Z-Score', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()

    return fig

