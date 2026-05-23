import re
import pandas as pd
from tqdm import tqdm

def clean_text(text: str) -> str:
    text = str(text).lower()
    # Hapus URL
    text = re.sub(r'http\S+|www\S+', '', text, flags=re.MULTILINE)
    # Hapus mention
    text = re.sub(r'@\w+', '', text)
    # Hapus hashtag
    text = re.sub(r'#\w+', '', text)
    # Hapus emoji dan tanda baca dengan hanya menyisakan huruf, angka, dan spasi
    # Ini penting agar kata yang menempel dengan tanda baca (misal: "bagus.") bisa terbaca oleh lexicon labeler
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    tqdm.pandas(desc="Preprocessing")
    
    # Sesuai instruksi: Jangan hapus satu kata pun (tanpa stopword) dan jangan ubah bentuk (tanpa stemming).
    # Hanya lakukan pembersihan dasar (URL, mention, hashtag, spasi berlebih, dan emoji).
    df['tweet_processed'] = df['tweet_raw'].progress_apply(clean_text)
    
    # Filter data kosong setelah dibersihkan
    before = len(df)
    df = df[df['tweet_processed'].str.strip() != ''].reset_index(drop=True)
    print(f"[filter_empty] Dropped {before - len(df)} tweets that became empty after preprocessing")
    
    return df
