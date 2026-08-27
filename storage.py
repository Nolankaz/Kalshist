from pathlib import Path
import pandas as pd

def save_daily(df, table_name, date):
    folder = Path("data") / table_name
    folder.mkdir(parents=True, exist_ok=True)

    file_path = folder / f"{date}.parquet"

    df.to_parquet(file_path, index=False)

def load_range(table_name, start_date, end_date):
    folder = Path("data") / table_name

    dates = pd.date_range(start=start_date, end=end_date)

    dataframes = []

    for date in dates:
        file_path = folder / f"{date.strftime('%Y-%m-%d')}.parquet"

        if file_path.exists():
            df = pd.read_parquet(file_path)
            dataframes.append(df)

    if not dataframes:
        return pd.DataFrame()

    return pd.concat(dataframes, ignore_index=True)