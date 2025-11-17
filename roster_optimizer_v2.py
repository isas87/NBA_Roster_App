import pandas as pd
import itertools
from typing import List, Dict, Any, Tuple, Optional, Set

# --- Helper Types ---
PlayerName = str
RosterList = List[PlayerName]
DailyLineup = Dict[str, List[PlayerName]]

# --- Column Name Constants (as function arguments) ---
DEFAULT_PLAYER_COL = 'player_name'
DEFAULT_POS_COL = 'position'
DEFAULT_COST_COL = 'current_cost'


def is_valid_roster(
        roster_names: RosterList,
        all_players_df: pd.DataFrame,
        budget: float,
        player_name_col: str = DEFAULT_PLAYER_COL,
        pos_col: str = DEFAULT_POS_COL,
        cost_col: str = DEFAULT_COST_COL
 ) -> bool:
    """
    Checks if a given 10-player roster is valid against budget and position constraints.
    """
    roster_df = all_players_df[all_players_df[player_name_col].isin(roster_names)]

    # 1. Check for 10 players
    if len(roster_df) != 10:
        # This can happen if a player name is invalid
        return False

    # 2. Check budget
    if roster_df[cost_col].sum() > budget:
        return False

    # 3. Check position counts (5 Backcourt 'BC', 5 Frontcourt 'FC')
    pos_counts = roster_df[pos_col].value_counts()
    if pos_counts.get('BC', 0) != 5 or pos_counts.get('FC', 0) != 5:
        return False

    return True


def optimize_daily_lineup(
        roster_df: pd.DataFrame,
        day_col: str,
        obj_var: str,
        pos_col: str = DEFAULT_POS_COL,
        player_name_col: str = DEFAULT_PLAYER_COL
) -> Tuple[float, RosterList]:
    """
    Finds the optimal 5-player lineup for a single day from a 10-player roster.
    Considers 3BC/2FC and 2BC/3FC combinations.
    """
    # Filter for players playing on the given day
    playing_df = roster_df[roster_df[day_col] == 1].sort_values(by=obj_var, ascending=False)

    bc_players = playing_df[playing_df[pos_col] == 'BC']
    fc_players = playing_df[playing_df[pos_col] == 'FC']

    best_score = 0.0
    best_lineup = []

    # Case 1: 3 Backcourt, 2 Frontcourt
    if len(bc_players) >= 3 and len(fc_players) >= 2:
        lineup_3bc_2fc = bc_players.head(3).append(fc_players.head(2))
        score_3bc_2fc = lineup_3bc_2fc[obj_var].sum()
        if score_3bc_2fc > best_score:
            best_score = score_3bc_2fc
            best_lineup = lineup_3bc_2fc[player_name_col].tolist()

    # Case 2: 2 Backcourt, 3 Frontcourt
    if len(bc_players) >= 2 and len(fc_players) >= 3:
        lineup_2bc_3fc = bc_players.head(2).append(fc_players.head(3))
        score_2bc_3fc = lineup_2bc_3fc[obj_var].sum()
        if score_2bc_3fc > best_score:
            best_score = score_2bc_3fc
            best_lineup = lineup_2bc_3fc[player_name_col].tolist()

    return best_score, best_lineup


def calculate_roster_weekly_score(
        roster_names: RosterList,
        all_players_df: pd.DataFrame,
        days: List[str],
        obj_var: str,
        pos_col: str = DEFAULT_POS_COL,
        player_name_col: str = DEFAULT_PLAYER_COL
) -> Tuple[float, DailyLineup]:
    """
    Calculates the total optimal score for a 10-player roster for an entire week.
    """
    roster_df = all_players_df[all_players_df[player_name_col].isin(roster_names)]
    if len(roster_df) != 10:
        return 0.0, {}  # Invalid roster

    total_weekly_score = 0.0
    daily_lineups = {}

    for day in days:
        daily_score, daily_lineup = optimize_daily_lineup(
            roster_df, day, obj_var, pos_col, player_name_col
        )
        total_weekly_score += daily_score
        daily_lineups[day] = daily_lineup

    return total_weekly_score, daily_lineups


def optimize_weekly_roster(
        starting_roster: RosterList,
        all_players_df: pd.DataFrame,
        budget: float,
        days: List[str],
        obj_var: str,
        player_name_col: str = DEFAULT_PLAYER_COL,
        pos_col: str = DEFAULT_POS_COL,
        cost_col: str = DEFAULT_COST_COL
) -> Tuple[RosterList, float, DailyLineup]:
    """
    Finds the best possible 10-player roster by making 0, 1, or 2 transfers
    from the starting_roster.
    """
    # Get player pools, excluding those already on the starting roster
    start_roster_set = set(starting_roster)
    all_players_df_excl_start = all_players_df[
        ~all_players_df[player_name_col].isin(start_roster_set)
    ]

    # Separate all available players by position
    all_bc_pool = all_players_df_excl_start[
        all_players_df_excl_start[pos_col] == 'BC'
        ][player_name_col].tolist()
    all_fc_pool = all_players_df_excl_start[
        all_players_df_excl_start[pos_col] == 'FC'
        ][player_name_col].tolist()

    # Separate starting roster by position
    start_roster_df = all_players_df[all_players_df[player_name_col].isin(start_roster_set)]
    start_bc_set = set(start_roster_df[start_roster_df[pos_col] == 'BC'][player_name_col])
    start_fc_set = set(start_roster_df[start_roster_df[pos_col] == 'FC'][player_name_col])

    best_roster = starting_roster
    best_score = -1.0
    best_daily_lineups = {}

    # --- 0 Transfers ---
    if is_valid_roster(starting_roster, all_players_df, budget, player_name_col, pos_col, cost_col):
        score, lineups = calculate_roster_weekly_score(
            starting_roster, all_players_df, days, obj_var, pos_col, player_name_col
        )
        best_score = score
        best_daily_lineups = lineups

    candidate_roster_sets: Set[Tuple[PlayerName, ...]] = set()

    # --- 1 Transfer ---
    # 1a: Drop 1 BC, Add 1 BC
    for p_out in start_bc_set:
        for p_in in all_bc_pool:
            new_bc = (start_bc_set - {p_out}) | {p_in}
            candidate_roster_sets.add(tuple(sorted(list(new_bc | start_fc_set))))

    # 1b: Drop 1 FC, Add 1 FC
    for p_out in start_fc_set:
        for p_in in all_fc_pool:
            new_fc = (start_fc_set - {p_out}) | {p_in}
            candidate_roster_sets.add(tuple(sorted(list(start_bc_set | new_fc))))

    # --- 2 Transfers ---
    # 2a: Drop 2 BC, Add 2 BC
    if len(start_bc_set) >= 2 and len(all_bc_pool) >= 2:
        for p_out_tuple in itertools.combinations(start_bc_set, 2):
            for p_in_tuple in itertools.combinations(all_bc_pool, 2):
                new_bc = (start_bc_set - set(p_out_tuple)) | set(p_in_tuple)
                candidate_roster_sets.add(tuple(sorted(list(new_bc | start_fc_set))))

    # 2b: Drop 2 FC, Add 2 FC
    if len(start_fc_set) >= 2 and len(all_fc_pool) >= 2:
        for p_out_tuple in itertools.combinations(start_fc_set, 2):
            for p_in_tuple in itertools.combinations(all_fc_pool, 2):
                new_fc = (start_fc_set - set(p_out_tuple)) | set(p_in_tuple)
                candidate_roster_sets.add(tuple(sorted(list(start_bc_set | new_fc))))

    # 2c: Drop 1 BC, 1 FC; Add 1 BC, 1 FC
    if len(all_bc_pool) >= 1 and len(all_fc_pool) >= 1:
        for p_out_bc in start_bc_set:
            for p_in_bc in all_bc_pool:
                for p_out_fc in start_fc_set:
                    for p_in_fc in all_fc_pool:
                        new_bc = (start_bc_set - {p_out_bc}) | {p_in_bc}
                        new_fc = (start_fc_set - {p_out_fc}) | {p_in_fc}
                        candidate_roster_sets.add(tuple(sorted(list(new_bc | new_fc))))

    # --- Evaluate all unique candidate rosters ---
    print(f"Evaluating {len(candidate_roster_sets)} unique candidate rosters...")

    for roster_tuple in candidate_roster_sets:
        roster_list = list(roster_tuple)

        # 1. Check validity
        if not is_valid_roster(roster_list, all_players_df, budget, player_name_col, pos_col, cost_col):
            continue

        # 2. Calculate score
        score, lineups = calculate_roster_weekly_score(
            roster_list, all_players_df, days, obj_var, pos_col, player_name_col
        )

        # 3. Update best
        if score > best_score:
            best_score = score
            best_roster = roster_list
            best_daily_lineups = lineups

    return best_roster, best_score, best_daily_lineups


def run_simulation(
        n_weeks: int,
        starting_roster: RosterList,
        weekly_dfs: List[pd.DataFrame],
        budget: float,
        days: List[str],
        obj_var: str,
        player_name_col: str = DEFAULT_PLAYER_COL,
        pos_col: str = DEFAULT_POS_COL,
        cost_col: str = DEFAULT_COST_COL
) -> List[Dict[str, Any]]:
    """
    Runs the multi-week optimization simulation.

    Args:
        n_weeks: Number of weeks to simulate.
        starting_roster: The 10-player roster for Week 1.
        weekly_dfs: A list of DataFrames, one for each week.
                      len(weekly_dfs) must be >= n_weeks.
                      Each df contains all players with their *predicted*
                      points and *costs* for that specific week.
        budget: The total salary cap.
        days: List of column names representing the days of the week.
        obj_var: Column name for predicted points to maximize.
        ... other column names

    Returns:
        A list of dictionaries, where each dict contains the
        results for one simulated week.
    """
    if len(weekly_dfs) < n_weeks:
        raise ValueError(f"Not enough DataFrames provided. Need {n_weeks}, got {len(weekly_dfs)}.")

    simulation_results = []
    current_roster = starting_roster

    for i in range(n_weeks):
        print(f"\n--- Simulating Week {i + 1} ---")
        current_week_df = weekly_dfs[i]

        # Check that starting roster is valid for the new week's data
        # (e.g. costs may have changed)
        if not is_valid_roster(current_roster, current_week_df, budget, player_name_col, pos_col, cost_col):
            print(f"Warning: Week {i + 1} starting roster is invalid (likely due to cost changes).")
            # If invalid, we must make a transfer.
            # The optimizer will handle this, as the 0-transfer
            # option will have a score of -1.0.

        optimized_roster, week_score, daily_lineups = optimize_weekly_roster(
            starting_roster=current_roster,
            all_players_df=current_week_df,
            budget=budget,
            days=days,
            obj_var=obj_var,
            player_name_col=player_name_col,
            pos_col=pos_col,
            cost_col=cost_col
        )

        transfers_made = set(optimized_roster) - set(current_roster)
        transfers_out = set(current_roster) - set(optimized_roster)

        print(f"Week {i + 1} Complete.")
        print(f"  Best Roster: {optimized_roster}")
        print(f"  Total Points: {week_score:.2f}")
        print(f"  Transfers In: {list(transfers_made)}")
        print(f"  Transfers Out: {list(transfers_out)}")

        result = {
            "week": i + 1,
            "roster": optimized_roster,
            "total_points": week_score,
            "daily_lineups": daily_lineups,
            "transfers_in": list(transfers_made),
            "transfers_out": list(transfers_out)
        }
        simulation_results.append(result)

        # The optimized roster becomes the starting roster for the next week
        current_roster = optimized_roster

    return simulation_results


# --- Example Usage ---
if __name__ == "__main__":

    # 1. Define parameters
    BUDGET = 60000
    OBJ_VAR = 'predicted_points'
    DAYS_OF_WEEK = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    N_WEEKS_TO_SIMULATE = 2

    # 2. Create sample data for all players
    # (In a real scenario, you would load this from a CSV/DB)
    data = {
        'player_name': [
            # 5 Starting BC
            'BC_Star_1', 'BC_Star_2', 'BC_Mid_1', 'BC_Scrub_1', 'BC_Scrub_2',
            # 5 Starting FC
            'FC_Star_1', 'FC_Star_2', 'FC_Mid_1', 'FC_Scrub_1', 'FC_Scrub_2',
            # 4 Transfer Options (BC)
            'BC_Option_1', 'BC_Option_2', 'BC_Option_3', 'BC_Option_4_High_Pts',
            # 4 Transfer Options (FC)
            'FC_Option_1', 'FC_Option_2', 'FC_Option_3', 'FC_Option_4_High_Pts',
        ],
        'position': [
            'BC', 'BC', 'BC', 'BC', 'BC',
            'FC', 'FC', 'FC', 'FC', 'FC',
            'BC', 'BC', 'BC', 'BC',
            'FC', 'FC', 'FC', 'FC',
        ],
        'current_cost': [
            11000, 10500, 8000, 4500, 4000,  # Start BC
            11500, 10000, 8500, 4000, 3500,  # Start FC
            9000, 7000, 5000, 9500,  # Options BC
            9200, 7100, 5100, 9800  # Options FC
        ],
        'predicted_points': [
            55, 52, 40, 20, 18,  # Start BC
            58, 50, 42, 19, 15,  # Start FC
            45, 35, 25, 60,  # Options BC (Option 4 is great)
            46, 36, 26, 62  # Options FC (Option 4 is great)
        ],
        # Define a weekly schedule (1 = playing, 0 = not playing)
        'Mon': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        'Tue': [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
        'Wed': [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        'Thu': [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        'Fri': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        'Sat': [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
        'Sun': [1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0],
    }

    # Create the DataFrame
    sample_df = pd.DataFrame(data)

    # For this example, we'll use the *same* DataFrame for both weeks.
    # In a real run, you would have a list of different DataFrames,
    # e.g., [df_week_1, df_week_2]
    all_weekly_dfs = [sample_df, sample_df]

    # 3. Define the starting roster for Week 1
    # (Must be 5 BC and 5 FC, and within budget)
    week_1_starting_roster = [
        'BC_Star_1', 'BC_Star_2', 'BC_Mid_1', 'BC_Scrub_1', 'BC_Scrub_2',
        'FC_Star_1', 'FC_Star_2', 'FC_Mid_1', 'FC_Scrub_1', 'FC_Scrub_2',
    ]

    # Check cost of starting roster
    start_cost = sample_df[
        sample_df['player_name'].isin(week_1_starting_roster)
    ]['current_cost'].sum()
    print(f"Starting Roster Cost: {start_cost} (Budget: {BUDGET})")

    if start_cost > BUDGET:
        print("Error: Starting roster is over budget!")
    else:
        # 4. Run the simulation
        results = run_simulation(
            n_weeks=N_WEEKS_TO_SIMULATE,
            starting_roster=week_1_starting_roster,
            weekly_dfs=all_weekly_dfs,
            budget=BUDGET,
            days=DAYS_OF_WEEK,
            obj_var=OBJ_VAR
        )

        # 5. Print final results
        print("\n--- Simulation Finished ---")
        for week_result in results:
            print(f"--- Week {week_result['week']} ---")
            print(f"  Total Points: {week_result['total_points']:.2f}")
            print(f"  Transfers In: {week_result['transfers_in']}")
            print(f"  Transfers Out: {week_result['transfers_out']}")
            print(f"  Final Roster: {week_result['roster']}")
            # print(f"  Daily Lineups: {week_result['daily_lineups']}") # Uncomment for full detail