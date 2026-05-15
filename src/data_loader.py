import pandas as pd
from src.config import DATA_RAW, PROVIDERS

def load_provider(provider: str) -> pd.DataFrame:
    file_path = DATA_RAW / f"data_{provider}.csv"
    df = pd.read_csv(file_path, lineterminator='\n')
    
    lengths = []
    for col in df.columns:
        s = df[col].dropna().astype(str)
        if len(s) == 0:
            continue
        if s.str.startswith('http').mean() > 0.5:
            continue
        if s.str.isnumeric().mean() > 0.5:
            continue
        if 'href' in col.lower() or 'src' in col.lower():
            continue
        
        mean_len = s.str.len().mean()
        lengths.append((col, mean_len))
        
    lengths.sort(key=lambda x: x[1], reverse=True)
    
    col1 = lengths[0][0]
    col2 = lengths[1][0] if len(lengths) > 1 else col1
    
    df = df.rename(columns={col1: 'tweet_part1', col2: 'tweet_part2'})
    df['provider'] = provider
    
    return df[['provider', 'tweet_part1', 'tweet_part2']]

def build_tweet_raw(df: pd.DataFrame) -> pd.DataFrame:
    part1 = df['tweet_part1'].fillna('').astype(str)
    part2 = df['tweet_part2'].fillna('').astype(str)
    
    df['tweet_raw'] = part1 + ' ' + part2
    df = df[df['tweet_raw'].str.strip() != '']
    return df

def load_all() -> pd.DataFrame:
    dfs = []
    for provider in PROVIDERS:
        df_p = load_provider(provider)
        dfs.append(df_p)
        
    df_combined = pd.concat(dfs, ignore_index=True)
    df_combined = build_tweet_raw(df_combined)
    return df_combined.reset_index(drop=True)
