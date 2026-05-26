import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POS_FILE = ROOT / "data/lexicon/positive_words.txt"
NEG_FILE = ROOT / "data/lexicon/negative_words.txt"

EXPLICIT_POS = {"bagus", "baik", "mantap", "kece", "cepat", "lancar", "murah", "stabil", "jernih", "kuat", "promo", "hadiah", "bonus", "mudah", "ramah", "keren", "terbaik", "mending", "lumayan", "sip", "mantab", "jaya"}
EXPLICIT_NEG = {"lambat", "lelet", "lemot", "mahal", "mati", "hilang", "jelek", "buruk", "hancur", "susah", "sulit", "bapuk", "ngeleg", "ngelag", "error", "eror", "gangguan", "kendala", "kecewa", "komplain", "nyedot", "sedot", "babi", "anjing", "tahi", "tai", "tolol", "goblok", "bodoh", "parah", "payah", "capek"}

def main():
    # Load all words
    pos_lines = []
    pos_words = set()
    with open(POS_FILE, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            w = line.strip().split('\t')[0].lower()
            pos_lines.append((w, line))
            if w and w != "word":
                pos_words.add(w)
                
    neg_lines = []
    neg_words = set()
    with open(NEG_FILE, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            w = line.strip().split('\t')[0].lower()
            neg_lines.append((w, line))
            if w and w != "word":
                neg_words.add(w)

    overlap = pos_words.intersection(neg_words)
    
    new_pos_lines = []
    new_neg_lines = []
    
    # Process positive
    for w, line in pos_lines:
        if w in overlap:
            if w in EXPLICIT_POS:
                new_pos_lines.append(line)
            # If explicit neg or ambiguous, we drop it from pos
        else:
            # Drop from pos if it's explicitly negative but somehow only in pos
            if w not in EXPLICIT_NEG:
                new_pos_lines.append(line)

    # Process negative
    for w, line in neg_lines:
        if w in overlap:
            if w in EXPLICIT_NEG:
                new_neg_lines.append(line)
            # If explicit pos or ambiguous, we drop it from neg
        else:
            if w not in EXPLICIT_POS:
                new_neg_lines.append(line)

    with open(POS_FILE, "w", encoding="utf-8") as f:
        for line in new_pos_lines:
            f.write(line)
            
    with open(NEG_FILE, "w", encoding="utf-8") as f:
        for line in new_neg_lines:
            f.write(line)
            
    print(f"Cleaned! Overlapping words handled: {len(overlap)}")

if __name__ == "__main__":
    main()