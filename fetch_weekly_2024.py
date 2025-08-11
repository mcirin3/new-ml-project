import pandas as pd

def fetch_nfl_savant_weekly(year: int, week: int) -> pd.DataFrame:
    url = f"https://nflsavant.com/stats.php?year={year}&week={week}&position=all"
    dfs = pd.read_html(url)  # returns list of DataFrames
    if not dfs:
        raise ValueError(f"No tables found for {year} W{week}")
    df = dfs[0]  # first table is usually the main stats table
    # Clean column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df

# Example usage:
week1_df = fetch_nfl_savant_weekly(2025, 1)
print(week1_df.head())
