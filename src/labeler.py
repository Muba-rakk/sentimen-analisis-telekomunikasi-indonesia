import pandas as pd
from src.config import LEXICON

def load_lexicon() -> tuple[set[str], set[str]]:
    pos_words: set[str] = set()
    with (open(LEXICON / "positive_words.txt", "r", encoding="utf-8", errors="replace") as f):
        for line in f:
            w = line.strip().split('\t')[0]
            if w and w.lower() != "word":
                pos_words.add(w.lower())

    neg_words: set[str] = set()
    with (open(LEXICON / "negative_words.txt", "r", encoding="utf-8", errors="replace") as f):
        for line in f:
            w = line.strip().split('\t')[0]
            if w and w.lower() != "word":
                neg_words.add(w.lower())

    return pos_words, neg_words

def count_sentiment_words(text: str, pos_words: set[str], neg_words: set[str]) -> tuple[int, int]:
    words = text.split()
    pos_count = sum(1 for word in words if word in pos_words)
    neg_count = sum(1 for word in words if word in neg_words)
    return pos_count, neg_count

def assign_label(pos_count: int, neg_count: int) -> int:
    return 1 if pos_count > neg_count else 0

def run_labeling(df: pd.DataFrame) -> pd.DataFrame:
    pos_words, neg_words = load_lexicon()
    
    pos_counts = []
    neg_counts = []
    
    for text in df['tweet_processed']:
        pos, neg = count_sentiment_words(str(text), pos_words, neg_words)
        pos_counts.append(pos)
        neg_counts.append(neg)
    
    df = df.copy()
    df['pos_count'] = pos_counts
    df['neg_count'] = neg_counts
    
    initial_rows = len(df)
    
    df = df[(df['pos_count'] > 0) | (df['neg_count'] > 0)].copy()
    
    dropped_rows = initial_rows - len(df)
    print(f"Dropped {dropped_rows} unlabelable rows (pos_count=0 and neg_count=0).")
    
    df['label'] = df.apply(lambda row: assign_label(row['pos_count'], row['neg_count']), axis=1)
    
    return df
