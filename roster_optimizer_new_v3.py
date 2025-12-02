# import pandas as pd
# import numpy as np
from itertools import combinations
from typing import List, Dict, Tuple, Optional
from data_loading import *
import streamlit as st

@st.cache_data
def pre_select_options(df: pd.DataFrame,
                       n_weeks: int,
                       min_rank_pts: int,
                       min_rank_scr: int,
                       starting_roster: List[str],
                       days:List[str]) -> pd.DataFrame:

    today = pd.Timestamp.today().normalize()
    date_plus = today + pd.Timedelta(weeks = n_weeks)
    days_until_sunday = (6 - date_plus.weekday() + 7) % 7  # In pandas: Monday=0, ..., Sunday=6
    days_until_sunday = 7 if days_until_sunday == 0 else days_until_sunday
    final_date = date_plus + pd.Timedelta(days=days_until_sunday)

    max_games = df['games_played'].max()

    df_filter = df[pd.to_datetime(df['when_back']) < final_date]  # exclude those that are not available
    df_filter = df_filter[df_filter['is_out'] < 2]  # exclude those that won't play for sure
    df_filter = df_filter[pd.to_datetime(df_filter['when_back']) < today]  # exclude those that won't be available
    df_filter = df_filter[df_filter['games_played'] / max_games > .7]
    df_filter = df_filter[
        ~((df_filter['rank_pts'] > min_rank_pts) | (df_filter['rank_ppc'] > 200) | (df_filter['rank_score'] > min_rank_scr))]

    if len(starting_roster) == 0:
        df_starting = pd.DataFrame()
    else:
        df_starting = df[df.player_name.isin(starting_roster)]  # Ensure the starting roster is in df_filter

    df_combined = pd.concat([df_filter, df_starting])  # , ignore_index = True)
    df_combined = df_combined.loc[~df_combined.index.duplicated(keep='first'), :]
    df_combined['games_available'] = df_combined[days].sum(axis=1)

    return df_combined

@st.cache_data
def get_week_days(days: List[str], week_num: int) -> List[str]:
    """
    Extract day columns for a specific week.
    """
    week_prefix = f"{week_num}_"
    return [day for day in days if day.startswith(week_prefix)]

@st.cache_data
def optimize_daily_lineup(roster_players: List[str], df: pd.DataFrame,
                          day_col: str, obj_var: str) -> Dict:
    """
    Optimize the 5-player lineup for a single day from a 10-player roster.
    """
    roster_df = df[df['player_name'].isin(roster_players)].copy()

    # Filter players who have games on this day
    available = roster_df[roster_df[day_col] == 1].copy()

    if len(available) < 5:
        # Not enough players available, return what we can
        return {
            'lineup': available['player_name'].tolist(),
            'total_points': available[obj_var].sum(),
            'backcourt': available[available['position'] == 'Backcourt']['player_name'].tolist(),
            'frontcourt': available[available['position'] == 'Frontcourt']['player_name'].tolist()
        }

    bc_available = available[available['position'] == 'Backcourt']
    fc_available = available[available['position'] == 'Frontcourt']

    best_lineup = None
    best_points = -1

    # Try 3BC + 2FC configuration
    if len(bc_available) >= 3 and len(fc_available) >= 2:
        for bc_combo in combinations(bc_available.index, 3):
            for fc_combo in combinations(fc_available.index, 2):
                points = (available.loc[list(bc_combo), obj_var].sum() +
                          available.loc[list(fc_combo), obj_var].sum())
                if points > best_points:
                    best_points = points
                    best_lineup = (list(bc_combo), list(fc_combo), '3BC-2FC')

    # Try 2BC + 3FC configuration
    if len(bc_available) >= 2 and len(fc_available) >= 3:
        for bc_combo in combinations(bc_available.index, 2):
            for fc_combo in combinations(fc_available.index, 3):
                points = (available.loc[list(bc_combo), obj_var].sum() +
                          available.loc[list(fc_combo), obj_var].sum())
                if points > best_points:
                    best_points = points
                    best_lineup = (list(bc_combo), list(fc_combo), '2BC-3FC')

    if best_lineup is None:
        return {
            'lineup': [],
            'total_points': 0,
            'backcourt': [],
            'frontcourt': [],
            'config': 'None'
        }

    bc_indices, fc_indices, config = best_lineup
    lineup_players = (available.loc[bc_indices, 'player_name'].tolist() +
                      available.loc[fc_indices, 'player_name'].tolist())

    return {
        'lineup': lineup_players,
        'total_points': best_points,
        'backcourt': available.loc[bc_indices, 'player_name'].tolist(),
        'frontcourt': available.loc[fc_indices, 'player_name'].tolist(),
        'config': config
    }

@st.cache_data
def evaluate_roster_week(roster_players: List[str], df: pd.DataFrame,
                         week_days: List[str], obj_var: str) -> Dict:
    """
    Evaluate a roster for an entire week by optimizing daily lineups.
    """
    daily_results = {}
    total_points = 0

    for day in week_days:
        day_result = optimize_daily_lineup(roster_players, df, day, obj_var)
        daily_results[day] = day_result
        total_points += day_result['total_points']

    return {
        'total_points': total_points,
        'daily_results': daily_results,
        'roster': roster_players
    }

@st.cache_data
def generate_wild_rosters(df: pd.DataFrame, budget: float, obj_var: str, 
                          candidates_per_pos: int = 15, max_results: int = 50) -> List[List[str]]:
    """
    Generate completely new rosters from scratch (Wild Mode).
    Selects top candidates per position, creates valid budget combinations,
    and returns the highest potential rosters for detailed evaluation.
    """
    # 1. Select top candidates to reduce search space (heuristic)
    # We take the top N players by objective variable (e.g., points)
    bc_candidates = df[df['position'] == 'Backcourt'].nlargest(candidates_per_pos, obj_var).copy()
    fc_candidates = df[df['position'] == 'Frontcourt'].nlargest(candidates_per_pos, obj_var).copy()
    
    # 2. Pre-calculate combinations for Backcourt (5 players)
    bc_combos = []
    for combo in combinations(bc_candidates.itertuples(), 5):
        cost = sum(p.current_cost for p in combo)
        pts = sum(getattr(p, obj_var) for p in combo)
        names = [p.player_name for p in combo]
        # Only keep if cost is theoretically viable (leaving room for min cost FC players)
        # Assuming min FC cost is roughly 5 * min_cost_in_pool, but checking strictly later is better
        bc_combos.append({'names': names, 'cost': cost, 'pts': pts})
        
    # 3. Pre-calculate combinations for Frontcourt (5 players)
    fc_combos = []
    for combo in combinations(fc_candidates.itertuples(), 5):
        cost = sum(p.current_cost for p in combo)
        pts = sum(getattr(p, obj_var) for p in combo)
        names = [p.player_name for p in combo]
        fc_combos.append({'names': names, 'cost': cost, 'pts': pts})
    
    # Sort combos by points descending to find best options first
    bc_combos.sort(key=lambda x: x['pts'], reverse=True)
    fc_combos.sort(key=lambda x: x['pts'], reverse=True)
    
    valid_rosters = []
    
    # 4. Merge BC and FC combos
    for bc in bc_combos:
        remaining_budget = budget - bc['cost']
        if remaining_budget < 0:
            continue
            
        for fc in fc_combos:
            if fc['cost'] <= remaining_budget:
                # Found a valid roster
                full_roster = bc['names'] + fc['names']
                valid_rosters.append(full_roster)
                
                # Limit the number of rosters to prevent timeouts
                if len(valid_rosters) >= max_results:
                    return valid_rosters
                    
    return valid_rosters

@st.cache_data
def generate_roster_swaps(starting_roster: List[str], df: pd.DataFrame,
                          max_swaps: int, budget: float) -> List[List[str]]:
    """
    Generate all valid roster combinations with up to max_swaps changes.
    """
    all_players = df['player_name'].tolist()
    available_players = [p for p in all_players if p not in starting_roster]

    # Validate starting roster is within budget
    if len(starting_roster) == 0:
        starting_cost = 0
        # If roster is empty but not using wild generator, standard logic might fail.
        # However, wild mode should handle empty starts.
    else:
        starting_cost = df[df['player_name'].isin(starting_roster)]['current_cost'].sum()

    valid_rosters = []

    # Only include starting roster if it's within budget
    if starting_cost <= budget:
        valid_rosters.append(starting_roster)

    # Generate rosters with 1 to max_swaps changes
    for num_swaps in range(1, max_swaps + 1):
        # Choose which players to remove
        for players_out in combinations(starting_roster, num_swaps):
            # Choose which players to add
            for players_in in combinations(available_players, num_swaps):
                new_roster = [p for p in starting_roster if p not in players_out]
                new_roster.extend(players_in)

                # Check constraints
                roster_df = df[df['player_name'].isin(new_roster)]

                # Budget constraint: Total cost of all 10 players must be <= budget
                total_roster_cost = roster_df['current_cost'].sum()
                if total_roster_cost > budget:
                    continue

                # Position constraints (5 BC, 5 FC)
                bc_count = (roster_df['position'] == 'Backcourt').sum()
                fc_count = (roster_df['position'] == 'Frontcourt').sum()

                if bc_count == 5 and fc_count == 5:
                    valid_rosters.append(new_roster)

    return valid_rosters

@st.cache_data
def optimize_roster_week(budget: float, starting_roster: List[str], df: pd.DataFrame,
                         week_days: List[str], obj_var: str = 'fg_pts',
                         max_swaps: int = 2, verbose: bool = True,
                         wildcard: bool = False) -> Dict:
    """
    Optimize roster for a single week.
    
    Args include 'wildcard': If True, ignores starting roster and generates fresh combinations.
    """
    if verbose:
        print(f"\nOptimizing roster for {len(week_days)} days...")
        if not wildcard:
            print(f"Starting roster cost: {df[df['player_name'].isin(starting_roster)]['current_cost'].sum():.0f}")
        else:
            print(f"WILD MODE: Generating fresh rosters from best available players...")

    # Generate valid roster combinations
    if wildcard:
        # Use the Wild logic to generate completely new rosters
        valid_rosters = generate_wild_rosters(df, budget, obj_var)
        st.info(f"WILD MODE: Evaluating {len(valid_rosters)} high-potential rosters...")
    else:
        # Use Standard logic (Swaps)
        valid_rosters = generate_roster_swaps(starting_roster, df, max_swaps, budget)
        st.info(f"Evaluating {len(valid_rosters)} valid roster combinations...")

    if verbose:
        print(f"Evaluating {len(valid_rosters)} valid roster combinations...")

    best_roster = None
    best_points = -1
    best_evaluation = None

    # Evaluate each roster
    for roster in valid_rosters:
        evaluation = evaluate_roster_week(roster, df, week_days, obj_var)

        if evaluation['total_points'] > best_points:
            best_points = evaluation['total_points']
            best_roster = roster
            best_evaluation = evaluation

    # Calculate changes from starting roster
    # In Wild mode, starting_roster is effectively ignored, so everything is technically "IN"
    # But to prevent UI clutter, if it's Wild, we can treat the starting roster as empty list for diffing
    reference_roster = [] if wildcard else starting_roster
    
    players_out = [p for p in reference_roster if p not in best_roster]
    players_in = [p for p in best_roster if p not in reference_roster]
    
    # If wild, we might not want to list 10 swaps, but for consistency we calculate them
    
    return {
        'optimal_roster': best_roster,
        'total_points': best_points,
        'players_out': players_out,
        'players_in': players_in,
        'num_swaps': len(players_in), # Use players_in count as 'swaps' for Wild mode consistency
        'daily_results': best_evaluation['daily_results'],
        'final_cost': df[df['player_name'].isin(best_roster)]['current_cost'].sum()
    }

@st.cache_data
def optimize_roster_multiweek(budget: float, starting_roster: List[str],
                              df: pd.DataFrame, start_week: int, num_weeks: int,
                              days: List[str], obj_var: str = 'fg_pts',
                              max_swaps: int = 2, verbose: bool = True,
                              wildcard: bool = False) -> Dict:
    """
    Optimize roster across multiple weeks.
    
    If wildcard is True, the logic will generate a completely fresh roster for the START week,
    and then use standard swaps for subsequent weeks.
    """
    weekly_results = {}
    current_roster = starting_roster.copy()
    total_points_all_weeks = 0

    end_week = start_week + num_weeks - 1

    # Validate initial roster
    initial_cost = df[df['player_name'].isin(starting_roster)]['current_cost'].sum()

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"MULTI-WEEK OPTIMIZATION: Weeks {start_week} to {end_week} ({num_weeks} weeks)")
        print(f"Budget: ${budget:.0f} | Max swaps per week: {max_swaps}")
        if wildcard:
            print("OPTION: WILD MODE ENABLED (Best 10 players selected for Week 1)")
        else:
            print(f"Initial roster cost: ${initial_cost:.0f}")
            if initial_cost > budget:
                print(f"WARNING: Initial roster exceeds budget by ${initial_cost - budget:.0f}!")
        print(f"{'=' * 70}")

    for i, week_num in enumerate(range(start_week, start_week + num_weeks)):
        if verbose:
            print(f"\n{'*' * 70}")
            print(f"WEEK {week_num}")
            print(f"{'*' * 70}")

        week_days = get_week_days(days, week_num)

        if not week_days:
            if verbose:
                print(f"Warning: No days found for week {week_num}")
            continue

        # Apply wildcard logic ONLY for the very first week of the loop if requested
        is_wild_week = wildcard and (i == 0)

        week_result = optimize_roster_week(
            budget=budget,
            starting_roster=current_roster,
            df=df,
            week_days=week_days,
            obj_var=obj_var,
            max_swaps=max_swaps,
            verbose=verbose,
            wildcard=is_wild_week
        )

        # Check if there was an error or empty result
        if not week_result.get('optimal_roster'):
             if verbose: print(f"Optimization failed or returned no roster for week {week_num}")
             break

        weekly_results[f'week_{week_num}'] = week_result
        total_points_all_weeks += week_result['total_points']

        # Print week summary
        if verbose:
            print(f"\n--- Week {week_num} Summary ---")
            print(f"Total Points: {week_result['total_points']:.1f}")
            print(f"Roster Cost: ${week_result['final_cost']:.0f} / ${budget:.0f}")

            if week_result['num_swaps'] > 0:
                print(f"\nRoster Changes ({week_result['num_swaps']} changes):")
                if is_wild_week:
                    print("  * WILD CARD USED: Complete Roster Reset *")
                
                # Limit output if it's a huge list (Wild Card)
                if is_wild_week and len(week_result['players_in']) > 5:
                     print(f"  (New roster selected with {len(week_result['players_in'])} players)")
                else:
                    for i, (out, in_) in enumerate(zip(week_result['players_out'], week_result['players_in']), 1):
                        out_info = df[df['player_name'] == out]
                        in_info = df[df['player_name'] == in_]
                        
                        # Handle potential missing data safely
                        if not out_info.empty:
                            out_cost = out_info['current_cost'].values[0]
                            out_pts = out_info[obj_var].values[0]
                        else:
                             out_cost, out_pts = 0, 0
                             
                        if not in_info.empty:
                            in_cost = in_info['current_cost'].values[0]
                            in_pts = in_info[obj_var].values[0]
                        else:
                             in_cost, in_pts = 0, 0

                        cost_diff = in_cost - out_cost
                        pts_diff = in_pts - out_pts
                        cost_sign = "+" if cost_diff > 0 else ""
                        pts_sign = "+" if pts_diff > 0 else ""
                        print(
                            f"  Swap {i}: {out} (${out_cost:.0f}, {out_pts:.1f}pts) -> {in_} (${in_cost:.0f}, {in_pts:.1f}pts)")
                        print(f"          Cost: {cost_sign}${cost_diff:.0f} | Points: {pts_sign}{pts_diff:.1f}")
            else:
                print("\nNo roster changes made (current roster is optimal)")

        # Update roster for next week
        current_roster = week_result['optimal_roster'].copy()

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"FINAL SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total Points (Weeks {start_week}-{end_week}): {total_points_all_weeks:.1f}")
        print(f"Average Points per Week: {total_points_all_weeks / num_weeks:.1f}")
        final_cost = df[df['player_name'].isin(current_roster)]['current_cost'].sum()
        print(f"Final Roster Cost: ${final_cost:.0f} / ${budget:.0f}")
        print(f"Remaining Budget: ${budget - final_cost:.0f}")
        print(f"Final Roster: {', '.join(current_roster)}")

    return {
        'weekly_results': weekly_results,
        'total_points': total_points_all_weeks,
        'final_roster': current_roster,
        'start_week': start_week,
        'end_week': end_week,
        'num_weeks': num_weeks
    }

@st.cache_data
def get_detailed_report(result: Dict, df: pd.DataFrame, obj_var: str = 'fg_pts'):
    """
    Generate a detailed report DataFrame from optimization results.
    """
    rows = []

    if 'weekly_results' in result:
        # Multi-week result
        for week_key, week_data in result['weekly_results'].items():
            week_num = week_key.split('_')[1]

            players_out = week_data.get('players_out', [])
            players_in = week_data.get('players_in', [])
            num_swaps = week_data.get('num_swaps', 0)

            # Check if this was a Wildcard reset (implied if swaps > 5 usually, or checking logic)
            # For report clarity, if num_swaps == 10, it's likely a fresh team or wild
            is_wild = num_swaps > 5

            if num_swaps > 0:
                if is_wild:
                    roster_changes_out = "Previous Roster Dropped"
                    roster_changes_in = "Wild Card Selection"
                    swap_details_str = "Complete Roster Reset (Wild Mode)"
                else:
                    roster_changes_out = ', '.join(players_out) if players_out else 'None'
                    roster_changes_in = ', '.join(players_in) if players_in else 'None'
                    
                    swap_details = []
                    for out, in_ in zip(players_out, players_in):
                        out_info = df[df['player_name'] == out]
                        in_info = df[df['player_name'] == in_]
                        
                        out_cost = out_info['current_cost'].values[0] if not out_info.empty else 0
                        in_cost = in_info['current_cost'].values[0] if not in_info.empty else 0
                        out_pts = out_info[obj_var].values[0] if not out_info.empty else 0
                        in_pts = in_info[obj_var].values[0] if not in_info.empty else 0
                        
                        swap_details.append(
                            f"{out} (${out_cost:.0f}, {out_pts:.1f}pts) → {in_} (${in_cost:.0f}, {in_pts:.1f}pts)")
                    swap_details_str = '; '.join(swap_details)
            else:
                roster_changes_out = 'No changes'
                roster_changes_in = 'No changes'
                swap_details_str = 'No swaps'

            for day, day_data in week_data['daily_results'].items():
                rows.append({
                    'week': week_num,
                    'day': day,
                    'num_swaps': num_swaps,
                    'players_removed': roster_changes_out,
                    'players_added': roster_changes_in,
                    'swap_details': swap_details_str,
                    'lineup_config': day_data.get('config', 'N/A'),
                    'players_active': ', '.join(day_data['lineup']),
                    'backcourt_active': ', '.join(day_data['backcourt']),
                    'frontcourt_active': ', '.join(day_data['frontcourt']),
                    'points': day_data['total_points']
                })
    else:
        # Single week result
        # (Logic is similar but simplified for single result dict)
        players_out = result.get('players_out', [])
        players_in = result.get('players_in', [])
        num_swaps = result.get('num_swaps', 0)
        is_wild = num_swaps > 5

        if num_swaps > 0:
            if is_wild:
                 roster_changes_out = "Previous Roster Dropped"
                 roster_changes_in = "Wild Card Selection"
                 swap_details_str = "Complete Roster Reset (Wild Mode)"
            else:
                roster_changes_out = ', '.join(players_out) if players_out else 'None'
                roster_changes_in = ', '.join(players_in) if players_in else 'None'
                swap_details = []
                for out, in_ in zip(players_out, players_in):
                     out_info = df[df['player_name'] == out]
                     in_info = df[df['player_name'] == in_]
                     out_cost = out_info['current_cost'].values[0] if not out_info.empty else 0
                     in_cost = in_info['current_cost'].values[0] if not in_info.empty else 0
                     out_pts = out_info[obj_var].values[0] if not out_info.empty else 0
                     in_pts = in_info[obj_var].values[0] if not in_info.empty else 0
                     swap_details.append(f"{out} (${out_cost:.0f}, {out_pts:.1f}pts) → {in_} (${in_cost:.0f}, {in_pts:.1f}pts)")
                swap_details_str = '; '.join(swap_details)
        else:
            roster_changes_out = 'No changes'
            roster_changes_in = 'No changes'
            swap_details_str = 'No swaps'

        for day, day_data in result['daily_results'].items():
            rows.append({
                'day': day,
                'num_swaps': num_swaps,
                'players_removed': roster_changes_out,
                'players_added': roster_changes_in,
                'swap_details': swap_details_str,
                'lineup_config': day_data.get('config', 'N/A'),
                'players_active': ', '.join(day_data['lineup']),
                'backcourt_active': ', '.join(day_data['backcourt']),
                'frontcourt_active': ', '.join(day_data['frontcourt']),
                'points': day_data['total_points']
            })

    df_rows = pd.DataFrame(rows)
    df_rows['week_points'] = df_rows.points.sum().round()
    df_lineup = df_rows[['week', 'day', 'lineup_config', 'players_active', 'backcourt_active', 'frontcourt_active', 'points']] if 'week' in df_rows else df_rows
    changes = df_rows[['week', 'num_swaps', 'players_removed', 'players_added', 'swap_details', 'week_points']] if 'week' in df_rows else df_rows

    return df_lineup, changes