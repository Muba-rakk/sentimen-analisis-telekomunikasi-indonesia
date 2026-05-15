import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tqdm import tqdm
import pandas as pd
from src.config import TWITTER_STOPWORDS

stemmer = StemmerFactory().create_stemmer()

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True, raise_on_error=False)

STOP_WORDS = set(stopwords.words('indonesian'))
STOP_WORDS.update(TWITTER_STOPWORDS)

def clean_text(text: str) -> str:
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove non-alpha chars (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str) -> list[str]:
    return text.lower().split()

def remove_stopwords(tokens: list[str]) -> list[str]:
    return [word for word in tokens if word not in STOP_WORDS]

def stem(tokens: list[str]) -> list[str]:
    return [stemmer.stem(word) for word in tokens]

def run_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    tqdm.pandas(desc="Preprocessing")
    
    def process(text):
        cleaned = clean_text(text)
        tokens = tokenize(cleaned)
        filtered_tokens = remove_stopwords(tokens)
        stemmed_tokens = stem(filtered_tokens)
        return " ".join(stemmed_tokens)
    
    df['tweet_processed'] = df['tweet_raw'].progress_apply(process)
    return df
