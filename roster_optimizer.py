import pandas as pd
import numpy as np
from pulp import LpProblem, LpVariable, LpMaximize, LpStatus, value, lpSum

def optimize_nba_roster(df: pd.DataFrame, days) -> pd.DataFrame:
    """
    Selects an optimal 10-player NBA roster subject to cost, position, and
    complex daily playing constraints, maximizing total predicted points.

    Args:
        df: DataFrame where the index contains the player name, and columns include
            'player_id', 'position', 'cost', 'points', and daily columns (e.g., 'Mon', 'Tue', ...).

    Returns:
        DataFrame containing the 10 selected players, including their names, or an empty DataFrame
        if no optimal solution is found.
    """
    if df.empty:
        print("Input DataFrame is empty.")
        return pd.DataFrame()

    # --- 1. Model Setup ---
    prob = LpProblem("NBA_Roster_Optimization", LpMaximize)

    # The original index contains the player name. Reset index to move the
    # player name into a column called 'player_name' for output and internal reference.
    # We use the existing index name or 'index' as the column to rename.

    index_col_name = df.index.name if df.index.name is not None else 'index'
    df = df.reset_index().rename(columns={index_col_name: 'player_name'}).copy()

    # Decision Variable: x_p = 1 if player p is selected, 0 otherwise
    players = df.index.tolist()
    x = LpVariable.dicts("Select", players, 0, 1, 'Binary')

    # Separate players by position and map indices to data (using 'BC' and 'FC')
    bc_indices = df[df['position'] == 'Backcourt'].index.tolist()
    fc_indices = df[df['position'] == 'Frontcourt'].index.tolist()

    non_day_cols = ['player_id', 'player_name', 'position', 'current_cost', 'fg_pts']

    # --- 2. Objective Function (Maximize Total Predicted Points) ---
    # Objective 5 is prioritized: Maximize the total predicted points for the week.
    prob += lpSum([df.loc[p, 'fg_pts'] * x[p] for p in players]), "Total_Predicted_Points"

    # --- 3. Hard Constraints ---

    # Constraint 1: Select exactly 10 players
    prob += lpSum([x[p] for p in players]) == 10, "C_Total_Players"

    # Constraint 2: Position Split (5 BC and 5 FC)
    prob += lpSum([x[p] for p in bc_indices]) == 5, "C_BC_Count"
    prob += lpSum([x[p] for p in fc_indices]) == 5, "C_FC_Count"

    # Constraint 3: Max Cost (Sum of costs <= 100)
    prob += lpSum([df.loc[p, 'current_cost'] * x[p] for p in players]) <= 100, "C_Total_Cost"

    # --- 4. Daily Position Constraint (3/2 OR 2/3) ---

    M = 5  # Big M constant, must be larger than the max count (3)

    # Auxiliary binary variable alpha_d: 1 for 3BC/2FC split, 0 for 2BC/3FC split
    alpha = LpVariable.dicts("Alpha", days, 0, 1, 'Binary')

    # Auxiliary binary variable beta_d: 1 if selected players play on this day, 0 otherwise
    beta = LpVariable.dicts("Beta", days, 0, 1, 'Binary')

    for day in days:
        # Calculate the number of selected BC and FC players playing on this day
        bc_playing_count = lpSum([df.loc[p, day] * x[p] for p in bc_indices])
        fc_playing_count = lpSum([df.loc[p, day] * x[p] for p in fc_indices])
        S_d = bc_playing_count + fc_playing_count  # Total selected players playing on day 'day'

        # Link beta_d (is_playing) to S_d (total playing count)
        # If S_d > 0, force beta_d = 1
        prob += S_d >= beta[day], f"C_Day_{day}_Beta_Lower"
        # If beta_d = 1, S_d <= M (always true)
        prob += S_d <= M * beta[day], f"C_Day_{day}_Beta_Upper"

        # Objective 4: Maximize the number of players in each day of the week
        # This is interpreted as: IF beta_d = 1 (at least one player plays),
        # THEN total players playing (S_d) MUST be 5 (3BC/2FC or 2BC/3FC).
        # Constraint: S_d = 5 OR beta_d = 0 (i.e., 5 - S_d <= M * (1 - beta_d))
        prob += 5 - S_d <= M * (1 - beta[day]), f"C_Day_{day}_Force_Five_Players"

        # Apply the 3BC/2FC OR 2BC/3FC split IF beta_d = 1
        # The constraints below force the required counts (3/2 or 2/3) if beta_d = 1.
        # If beta_d = 0, the right-hand side (RHS) of M becomes M, making the constraint non-binding (satisfied).

        # --- IF alpha_d = 1 (Target: BC=3, FC=2) ---
        # 1. BC count must be 3:
        prob += 3 - bc_playing_count <= M * (1 - alpha[day] + (1 - beta[day])), f"C_Day_{day}_BC_3_Upper"
        prob += bc_playing_count - 3 <= M * (1 - alpha[day] + (1 - beta[day])), f"C_Day_{day}_BC_3_Lower"
        # 2. FC count must be 2:
        prob += 2 - fc_playing_count <= M * (1 - alpha[day] + (1 - beta[day])), f"C_Day_{day}_FC_2_Upper"
        prob += fc_playing_count - 2 <= M * (1 - alpha[day] + (1 - beta[day])), f"C_Day_{day}_FC_2_Lower"

        # --- IF alpha_d = 0 (Target: BC=2, FC=3) ---
        # 3. BC count must be 2:
        prob += 2 - bc_playing_count <= M * (alpha[day] + (1 - beta[day])), f"C_Day_{day}_BC_2_Upper"
        prob += bc_playing_count - 2 <= M * (alpha[day] + (1 - beta[day])), f"C_Day_{day}_BC_2_Lower"
        # 4. FC count must be 3:
        prob += 3 - fc_playing_count <= M * (alpha[day] + (1 - beta[day])), f"C_Day_{day}_FC_3_Upper"
        prob += fc_playing_count - 3 <= M * (alpha[day] + (1 - beta[day])), f"C_Day_{day}_FC_3_Lower"

    # --- 5. Solve the Problem ---
    try:
        prob.solve()
    except Exception as e:
        print(f"PuLP solver error: {e}")
        return pd.DataFrame()

    # --- 6. Extract Results ---
    if LpStatus[prob.status] == "Optimal":
        selected_players = [i for i in players if value(x[i]) == 1.0]

        result_df = df.loc[selected_players].copy()
        result_df['player_id'] = result_df.index  # Add index back for clarity

        # Determine which columns to return, including 'player_name'
        output_cols = ['player_id', 'player_name', 'position', 'current_cost', 'fg_pts'] + days

        # Calculate counts for printing
        bc_count_final = result_df[result_df['position'] == 'Backcourt'].shape[0]
        fc_count_final = result_df[result_df['position'] == 'Frontcourt'].shape[0]

        print(f"Optimization Status: Optimal (Max Points: {value(prob.objective):.2f})")
        print(f"Total Cost of Roster: {result_df['current_cost'].sum()}")
        # print(f"Roster size: {len(result_df)} (BC: {bc_count_final}, FC: {fc_count_final})")
        # display(result_df.head())

        return result_df[output_cols]
    else:
        print(f"Optimization Status: {LpStatus[prob.status]}")
        print("No solution found that satisfies all the strict constraints.")
        return pd.DataFrame()

def run_optimization(budget, week_start, num_weeks, df_rank):
    len = week_start + num_weeks - 1

    # Generate the list of column names based on the start and end range
    week_columns = [
        f'{i}_{j}'
        for i in range(week_start, len + 1)
        for j in range(1, 7)
    ]

    df_rank['games_available'] = df_rank[week_columns].sum(axis=1)

    print("--- Starting Optimization ---")
    optimized_roster_df = optimize_nba_roster(df_rank, week_columns)

    if not optimized_roster_df.empty:
        print("\n--- Optimized Roster ---")
        print(optimized_roster_df)

        # Verify daily constraints on the optimal roster
        print("\n--- Daily Split Verification ---")
        for day in days_of_week:
            bc_count = \
            optimized_roster_df[(optimized_roster_df['position'] == 'Backcourt') & (optimized_roster_df[day] == 1)].shape[0]
            fc_count = \
            optimized_roster_df[(optimized_roster_df['position'] == 'Frontcourt') & (optimized_roster_df[day] == 1)].shape[
                0]
            total_count = bc_count + fc_count
            status = "OK (5 total, 3/2 or 2/3 split)" if total_count == 5 and ((bc_count == 3 and fc_count == 2) or (
                        bc_count == 2 and fc_count == 3)) else "N/A (Total playing != 5)" if total_count != 0 else "N/A (No players playing)"
            print(f"{day}: BC={bc_count}, FC={fc_count}, Total={total_count} -> {status}")