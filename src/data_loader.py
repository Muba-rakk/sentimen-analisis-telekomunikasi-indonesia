import pandas as pd
from src.config import DATA_RAW, PROVIDERS
import os

def load_new_data() -> pd.DataFrame:
    """Load newly scraped data from playstore and twitter."""
    files = ['data_playstore.csv', 'data_twitter.csv']
    dfs = []
    
    for file in files:
        file_path = DATA_RAW / file
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, lineterminator='\n')
                if not df.empty:
                    dfs.append(df)
            except pd.errors.EmptyDataError:
                print(f"Warning: {file} is empty. Skipping.")
        else:
            print(f"Warning: {file} not found.")
            
    if dfs:
        df_combined = pd.concat(dfs, ignore_index=True)
        # Ensure raw text is string
        df_combined['tweet_raw'] = df_combined['tweet_raw'].fillna('').astype(str)
        # Drop empty tweets
        df_combined = df_combined[df_combined['tweet_raw'].str.strip() != '']
        return df_combined
    else:
        # Fallback empty dataframe if no files found
        return pd.DataFrame(columns=['source', 'provider', 'date', 'username', 'tweet_raw'])

def filter_cs_replies(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out Customer Service reply tweets using metadata (author username)."""
    if 'username' in df.columns:
        usernames = df['username'].astype(str).str.lower()
        # Filter all official accounts
        cs_mask = usernames.str.contains('indosat|telkomsel|xlaxiata|myxl|xlcare', na=False)
        n_dropped = cs_mask.sum()
        print(f"[filter_cs_replies] Dropped {n_dropped} CS reply tweets by username")
        return df[~cs_mask].reset_index(drop=True)
    return df

def filter_spam(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out non-Indonesian (English/Turkish/etc) and spam/sales tweets."""
    spam_patterns = [
        r'\bwtb\b', r'\bwts\b', r'\bjasa cv\b', r'\bcv pulsa\b', r'\bconvert pulsa\b',
        r'\b(the|and|is|to|that|it|of|for|you|in|my|on|this|we|with|are|be|have|not)\b',
        r'\b(size|ver|venue|goods|shirt|tshirt|acrylic|stand|towel)\b',
        r'\b(trknet|türknet|türksat|turksat|fiber|altyapısı|sabit|kullandım|memnun|kaldım)\b'
    ]
    pattern = '|'.join(spam_patterns)
    mask = df['tweet_raw'].str.contains(pattern, case=False, na=False, regex=True)
    n_dropped = mask.sum()
    print(f"[filter_spam] Dropped {n_dropped} spam/foreign tweets")
    return df[~mask].reset_index(drop=True)

def filter_telecom_only(df: pd.DataFrame) -> pd.DataFrame:
    """Strictly keep ONLY tweets that contain telecommunication keywords."""
    telecom_keywords = [
        r'\bindosat\b', r'\bim3\b', r'\booredoo\b', r'\bm3\b', r'\btri\b', r'\bthree\b',
        r'\btelkomsel\b', r'\btsel\b', r'\bsimpati\b', r'\bhalo\b', r'\bbyu\b', r'\bloop\b',
        r'\bxl\b', r'\baxiata\b', r'\baxis\b',
        r'\bsinyal\b', r'\bkuota\b', r'\bpaket\b', r'\bjaringan\b', r'\binternet\b', r'\bwifi\b', r'\bkoneksi\b', 
        r'\bmbps\b', r'\bgb\b', r'\bprovider\b', r'\bbts\b', r'\blemot\b', r'\blancar\b', r'\bgangguan\b', r'\bspeed\b', r'\blelet\b', r'\bkouta\b'
    ]
    pattern = '|'.join(telecom_keywords)
    mask = df['tweet_raw'].str.contains(pattern, case=False, na=False, regex=True)
    n_dropped = (~mask).sum()
    print(f"[filter_telecom_only] Dropped {n_dropped} out-of-context tweets")
    return df[mask].reset_index(drop=True)

def load_all() -> pd.DataFrame:
    # We now exclusively use the new data sources
    df_combined = load_new_data()
    
    if df_combined.empty:
        print("No new data found! Please run the scrapers first.")
        return df_combined

    # Drop CS first by username
    df_combined = filter_cs_replies(df_combined)
    
    # Drop spam and non-Indonesian
    df_combined = filter_spam(df_combined)
    
    # Drop completely out of context tweets, but ONLY for social media
    # Playstore reviews are inherently about the provider app
    mask_socmed = df_combined['source'].isin(['twitter', 'facebook'])
    if mask_socmed.any():
        df_socmed = df_combined[mask_socmed]
        df_playstore = df_combined[~mask_socmed]
        df_socmed_filtered = filter_telecom_only(df_socmed)
        df_combined = pd.concat([df_playstore, df_socmed_filtered], ignore_index=True)
    
    before = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['tweet_raw']).reset_index(drop=True)
    print(f"[drop_duplicates] Dropped {before - len(df_combined)} duplicate tweets")
        
    return df_combined
