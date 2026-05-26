from typing import Any
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack as scipy_hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


def build_tfidf(
    X_train: pd.Series, X_test: pd.Series
) -> tuple[Any, Any, TfidfVectorizer]:
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train.astype(str))
    X_test_tfidf = vectorizer.transform(X_test.astype(str))
    return X_train_tfidf, X_test_tfidf, vectorizer


def build_lexicon_features(
    df: pd.DataFrame, train_idx, test_idx
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    features = df.loc[train_idx][['pos_count', 'neg_count']].values
    scaler = MinMaxScaler()
    lex_train_norm = scaler.fit_transform(features)
    lex_test_norm = scaler.transform(df.loc[test_idx][['pos_count', 'neg_count']].values)
    return lex_train_norm, lex_test_norm, scaler


def combine_features(
    X_tfidf: Any, lex_features: np.ndarray
) -> Any:
    return scipy_hstack([X_tfidf, csr_matrix(lex_features)], format='csr')
