import pandas as pd
import numpy as np
from scipy import stats

# Identify best and worst performances
def get_performance_extremes(df, player_name, metric, top_n=5):
    """Get top and bottom performances for a player"""
    player_df = df[df['player_name'] == player_name].sort_values('etl_date')
    best = player_df.nlargest(top_n, metric)[[metric]].max().iloc[0]
    worst = player_df.nsmallest(top_n, metric)[[metric]].min().iloc[0]

    return best, worst

# Analyze performance trends over time
def calculate_trend(df, players, metric):

    linear_trend_data = []
    """Calculate linear trend for a player's metric over time"""
    df['etl_date'] = pd.to_datetime(df['etl_date'])
    for player_name in players:
        player_df = df[df['player_name'] == player_name].sort_values(by = 'etl_date')

        if len(player_df) == 0:
            continue

        x = np.arange(len(player_df))
        y = player_df[metric].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        best, worst = get_performance_extremes(player_df, player_name, metric, top_n=5)

        linear_trend_data.append(
            [player_name, slope, r_value ** 2, p_value, 'improving' if slope > 0 else 'declining', best, worst])

    return pd.DataFrame(linear_trend_data,
                        columns=['player_name', 'slope', 'r2_value', 'p_value', 'trend', 'best', 'worst'])

# Calculate consistency score (coefficient of variation)
def calculate_consistency(df, players, metric):

    consistency_data = []

    for player_name in players:
        player_df = df[df['player_name'] == player_name]

        if len(player_df) == 0:
            continue

        player_scores = {'player_name': player_name}

        mean_val = player_df[metric].mean()
        std_val = player_df[metric].std()
        cv = (std_val / mean_val) * 100 if mean_val > 0 else 0
        player_scores[f'{metric}_consistency'] = cv

        consistency_data.append(player_scores)

    return pd.DataFrame(consistency_data)