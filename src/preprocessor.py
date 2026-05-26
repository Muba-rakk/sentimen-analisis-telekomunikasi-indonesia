import re
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from src.config import TWITTER_STOPWORDS, SLANG_DICT

# Inisialisasi Stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Inisialisasi Stopwords NLTK (Indonesian) + Custom
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('indonesian'))
stop_words.update(TWITTER_STOPWORDS)

def clean_text(text: str) -> str:
    text = str(text).lower()
    # Hapus URL
    text = re.sub(r'http\S+|www\S+', '', text, flags=re.MULTILINE)
    # Hapus mention
    text = re.sub(r'@\w+', '', text)
    # Hapus hashtag
    text = re.sub(r'#\w+', '', text)
    # Hapus emoji dan tanda baca dengan hanya menyisakan huruf abjad dan spasi
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text: str) -> str:
    tokens = text.split()
    filtered_tokens = [w for w in tokens if w not in stop_words]
    return " ".join(filtered_tokens)

def normalize_slang(text: str) -> str:
    tokens = text.split()
    normalized = [SLANG_DICT.get(w, w) for w in tokens]
    return " ".join(normalized)

def apply_stemming(text: str) -> str:
    return stemmer.stem(text)

def run_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    tqdm.pandas(desc="Preprocessing (Clean -> Stopwords -> Stemming)")
    
    def process_row(text):
        text = clean_text(text)
        text = normalize_slang(text)
        text = remove_stopwords(text)
        text = apply_stemming(text)
        return text
        
    df['tweet_processed'] = df['tweet_raw'].progress_apply(process_row)
    
    # Filter data kosong setelah dibersihkan
    before = len(df)
    df = df[df['tweet_processed'].str.strip() != ''].reset_index(drop=True)
    print(f"[filter_empty] Dropped {before - len(df)} tweets that became empty after preprocessing")
    
    return df
