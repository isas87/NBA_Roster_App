import pandas as pd
import numpy as np
from scipy import stats


# Analyze performance trends over time
def calculate_trend(df, player_name, metric):
    """Calculate linear trend for a player's metric over time"""
    player_df = df[df['player_name'] == player_name].sort_values('etl_date')

    if len(player_df) < 2:
        return None

    x = np.arange(len(player_df))
    y = player_df[metric].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    return {
        'slope': slope,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'trend': 'improving' if slope > 0 else 'declining'
    }


# Compare multiple players over time
def compare_players_timeseries(df, players, metric):
    """Create comparison dataframe for multiple players"""
    comparison_data = []

    for player in players:
        player_data = df[df['player_name'] == player][['etl_date', metric]].copy()
        player_data['player_name'] = player
        comparison_data.append(player_data)

    return pd.concat(comparison_data, ignore_index=True)


# Calculate period-over-period changes
def calculate_pop_change(df, player_name, periods=1):
    """Calculate period-over-period percentage change"""
    player_df = df[df['player_name'] == player_name].sort_values('etl_date')

    metrics = ['fg_pts','pts', 'ast', 'trb', 'blk', 'stl']

    for metric in metrics:
        player_df[f'{metric}_pct_change'] = player_df[metric].pct_change(periods=periods) * 100

    return player_df


# Identify best and worst performances
def get_performance_extremes(df, player_name, metric, top_n=5):
    """Get top and bottom N performances for a player"""
    player_df = df[df['player_name'] == player_name].sort_values('etl_date')

    best = player_df.nlargest(top_n, metric)[['etl_date', metric]]
    worst = player_df.nsmallest(top_n, metric)[['etl_date', metric]]

    return {'best': best, 'worst': worst}


# Calculate consistency score (coefficient of variation)
def calculate_consistency(df, player_name):
    """Calculate consistency score for each metric (lower is more consistent)"""
    player_df = df[df['player_name'] == player_name]

    consistency_scores = {}
    metrics = ['fg_pts','pts', 'ast', 'trb', 'blk', 'stl']

    for metric in metrics:
        mean_val = player_df[metric].mean()
        std_val = player_df[metric].std()
        cv = (std_val / mean_val) * 100 if mean_val > 0 else 0
        consistency_scores[metric] = cv

    return consistency_scores