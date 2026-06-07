from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_comparison(results_a: dict, results_b: dict, save_path: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Model A (TF-IDF)", "Model B (TF-IDF + Lexicon)"]
    
    model_a_vals = [results_a[m] for m in metrics]
    model_b_vals = [results_b[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, model_a_vals, width, label=labels[0])
    ax.bar(x + width / 2, model_b_vals, width, label=labels[1])
    
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: TF-IDF vs TF-IDF + Lexicon")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.legend()
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def evaluate_per_provider(model, vectorizer, scaler, df_test: pd.DataFrame, use_lexicon: bool = False) -> pd.DataFrame:
    results = []
    for provider in df_test["provider"].unique():
        df_p = df_test[df_test["provider"] == provider]
        X_p = vectorizer.transform(df_p["tweet_processed"].astype(str))
        if use_lexicon:
            lex_p = df_p[["pos_count", "neg_count"]].values
            from scipy.sparse import csr_matrix, hstack as scipy_hstack
            lex_p_norm = scaler.transform(lex_p)
            X_p = scipy_hstack([X_p, csr_matrix(lex_p_norm)], format='csr')
        y_true_p = df_p["label"].values
        y_pred_p = model.predict(X_p)
        acc = accuracy_score(y_true_p, y_pred_p)
        results.append({"provider": provider, "accuracy": acc})
    return pd.DataFrame(results)


def save_results(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([results])
    df.to_csv(path, index=False)
