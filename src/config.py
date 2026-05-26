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

# Stopwords khusus bahasa gaul/Twitter yang sering muncul di review
TWITTER_STOPWORDS = [
    "rt", "yg", "dg", "dgn", "nya", "kak", "gan",
    "bgt", "banget", "aja", "udah", "udh", "gw",
    "gue", "lo", "lu", "nih", "sih", "dong", "deh",
    "kalo", "klo", "tp", "tapi", "yg", "aja", "jd",
    "sy", "aku", "kamu", "dia", "g", "gk", "ga", "nda",
    "ini", "itu", "ke", "di", "dari", "yaa", "ya"
]

SLANG_DICT = {
    "ngeleg": "lambat",
    "ngelag": "lambat",
    "ngelunjak": "kurang ajar",
    "kouta": "kuota",
    "ajgg": "anjing",
    "taii": "tahi",
    "bngt": "sangat",
    "sndri": "sendiri",
    "bapuk": "jelek",
    "cpt": "cepat",
    "hbis": "habis",
    "tetep": "tetap",
    "benerin": "perbaiki",
    "blakangan": "belakangan",
    "kmren": "kemarin",
    "karna": "karena",
    "krn": "karena",
    "jga": "juga",
    "jg": "juga",
    "mw": "mau",
    "trs": "terus",
    "gak": "tidak",
    "gk": "tidak",
    "ga": "tidak",
    "nda": "tidak"
}
