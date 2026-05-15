from pathlib import Path

ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_RAW   = ROOT_DIR / "data" / "raw"
DATA_PROC  = ROOT_DIR / "data" / "processed"
LEXICON    = ROOT_DIR / "data" / "lexicon"
RESULTS    = ROOT_DIR / "results"
FIGURES    = ROOT_DIR / "results" / "figures"

RANDOM_STATE = 42
TEST_SIZE    = 0.3
LR_PARAMS    = {"solver": "lbfgs", "max_iter": 1000, "random_state": RANDOM_STATE}
TUNING_C     = [0.01, 0.1, 1, 10, 100]

PROVIDERS = ["indosat", "telkomsel", "xl"]

TWITTER_STOPWORDS = [
    "rt", "yg", "dg", "dgn", "nya", "kak", "gan",
    "bgt", "banget", "aja", "udah", "udh", "gw",
    "gue", "lo", "lu", "nih", "sih", "dong", "deh"
]
