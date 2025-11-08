import streamlit as st
import pandas as pd

@st.cache_data
def scrape_current_data():
    """
    Scrapes the NBA 'Per Game' statistics for the 2025 season from Basketball Reference.
    The function uses st.info/st.error for status updates in the Streamlit environment.
    """
    url = 'https://www.basketball-reference.com/leagues/NBA_2026_per_game.html'

    try:
        # pd.read_html automatically finds tables on the webpage
        data = pd.read_html(url)
    except Exception as e:
        st.error(f"Error scraping data from Basketball Reference. Check URL or internet connection. Error: {e}")
        # Fallback to an empty DataFrame if scraping fails
        return pd.DataFrame({
            'Rk':[], 'Player': [], 'Age': [], 'Team': [], 'Pos': [], 'G': [], 'GS': [], 'MP': [], 'FG': [],
            'FGA': [], 'FG%':[], '3P':[], '3PA':[], '3P%':[], '2P':[], '2PA':[], '2P%':[], 'eFG%':[], 'FT':[],
            'FTA':[], 'FT%': [], 'ORB': [], 'DRB': [], 'TRB': [], 'AST': [], 'STL': [], 'BLK': [], 'TOV': [],
            'PF': [], 'PTS': [], 'Awards': []
        })

    # The main player stats table is usually the first one found
    df = data[0]

    # Data Cleaning and Filtering
    # 1. Filter out repeated header rows within the table (where 'Player' equals 'Player')
    df = df[df['Player'] != 'Player'].copy()
    df = df[~df['Player'].str.contains('Average')].copy()

    # 2.1 Rename columns for easier maintenance and normalize players names
    df = df.rename(
        columns={'Rk': 'week_rank', 'Player': 'player_name', 'Age': 'player_age', 'Team': 'team',
                 'Pos': 'position','G': 'games_played', 'GS': 'games_started', 'MP': 'minutes_played',
                 'FG': 'fg', 'FGA': 'fga', 'FG%': 'fg_pcnt', '3P': '3p', '3PA': '3pa', '3P%': '3p_pcnt',
                 '2P': '2p', '2PA': '2pa', '2P%': '2p_pcnt', 'eFG%': 'efg_pnt', 'FT': 'ft', 'FTA': 'fta',
                 'FT%': 'ft_pcnt', 'ORB': 'orb', 'DRB': 'drb', 'TRB': 'trb', 'AST': 'ast', 'STL': 'stl',
                 'BLK': 'blk', 'TOV': 'tov', 'PF': 'pf', 'PTS': 'pts', 'Awards': 'awards'}
        )

    # 2.2 Handle players names and unusual characters
    pattern_c = '|'.join(['ć', 'č'])
    pattern_e = '|'.join(['é'])
    df['player_name'] = df['player_name'].str.replace(pattern_c, 'c', regex=True)
    df['player_name'] = df['player_name'].str.replace(pattern_e, 'e', regex=True)

    # 3. Handle players with more than 1 team
    # Select players that played for more than 1 team
    df_2TM = df.loc[df.team.isin(['2TM', '3TM'])].copy()

    # Remove player with more than 1 team
    df_aux = df.loc[~(df.player_name.isin(df_2TM.player_name))].copy()

    # Select data for player with 1 team
    df_aux = df_aux.loc[df_aux.player_name.isna() == False]

    # concat data
    df = pd.concat([df_aux, df_2TM], axis=0)

    # Normalize Players positions
    df['position'] = df['position'].apply(
        lambda x: 'Frontcourt' if x in ['C', 'PF', 'SF', 'C-PF', 'PF-C', 'SF-PF'] else 'Backcourt'
    )

    # 4. Create main stat column (fantasy points)
    df['fg_pts'] = df['pts'] + df['trb'] + 2*df['ast'] + 3*df['blk'] + 3*df['stl']

    # 5. Remove players with low number of minutes played
    df = df[df.minutes_played > 5].copy()

    # 6. Select relevant columns for 9-Category H2H fantasy scoring
    cols_to_keep = ['player_name','pts', 'trb', 'ast', 'stl', 'blk', 'tov', 'games_played',
                    'minutes_played', 'games_started', 'fg', 'fga', 'fg_pcnt', '3p', '3pa', '3p_pcnt', '2p', '2pa',
                    '2p_pcnt', 'efg_pnt','fg_pts']

    # Ensure the required columns exist before selecting
    missing_cols = [col for col in cols_to_keep if col not in df.columns]

    if missing_cols:
        st.warning(f"The following columns were missing after scraping: {missing_cols}. Using available data.")
        # Attempt to proceed with available columns, but for this specific source, they should exist.
        cols_to_keep = [col for col in cols_to_keep if col in df.columns]

    df = df[cols_to_keep]

    # 3. Convert statistical columns to numeric
    stats_cols = [col for col in cols_to_keep if col != 'player_name']

    for col in stats_cols:
        # Coerce errors to NaN for players who might have null values
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Drop rows where key stats are NaN (e.g., players with very little play time)
    df = df.dropna(subset=stats_cols)

    # st.success(f"Successfully scraped data for {len(df)} players.")
    # st.info(f"Data scraped from: {url}")

    return df

@st.cache_data
def load_cost_data():
    # Transformed URL for CSV export of the sheet with gid=1056607967
    GSHEET_COST_URL = "https://docs.google.com/spreadsheets/d/15tiGXPRU1jYF_4I2ee6U3CrRBZecUgoq5N7HX_DupB4/gviz/tq?tqx=out:csv&gid=1056607967"
    # st.info(f"Attempting to load cost data from Google Sheet...")

    try:
        df = pd.read_csv(GSHEET_COST_URL)
        df = df[['player_name', 'team', 'current_cost', 'current_selection', 'current_form', 'current_total_points']]
        return df

    except Exception as e:
        st.error(
            f"Error loading cost data from Google Sheet. Ensure the sheet is published to the web or the URL is correct. Error: {e}")
        # Fallback to an empty DataFrame if loading fails
        return pd.DataFrame({
            'player_name': [], 'team': [], 'current_cost': [], 'current_selection': [], 'current_form': [],
            'current_total_points': []
        })

@st.cache_data
def load_schedule_data():
    GSHEET_SCHED_URL = "https://docs.google.com/spreadsheets/d/15tiGXPRU1jYF_4I2ee6U3CrRBZecUgoq5N7HX_DupB4/gviz/tq?tqx=out:csv&gid=169891812"

    try:
        df_sched = pd.read_csv(GSHEET_SCHED_URL)
        df_sched = df_sched.fillna(0)
        return df_sched
    except Exception as e:
        st.error(
            f"Error loading scheduled data from Google Sheet. Ensure the sheet is published to the web or the URL is correct. Error: {e}"
            )
