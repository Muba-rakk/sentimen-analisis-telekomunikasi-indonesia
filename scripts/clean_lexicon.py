import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POS_FILE = ROOT / "data/lexicon/positive_words.txt"
NEG_FILE = ROOT / "data/lexicon/negative_words.txt"

def main():
    # Load positive words
    pos_words = set()
    with open(POS_FILE, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            w = line.strip().split('\t')[0]
            if w and w.lower() != "word":
                pos_words.add(w.lower())
    
    # Filter negative words
    valid_neg_lines = []
    dropped = 0
    with open(NEG_FILE, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            w = line.strip().split('\t')[0]
            if w.lower() in pos_words:
                dropped += 1
            else:
                valid_neg_lines.append(line)
    
    # Save back
    with open(NEG_FILE, "w", encoding="utf-8") as f:
        for line in valid_neg_lines:
            f.write(line)
            
    print(f"Lexicon cleaned! Removed {dropped} overlapping positive words from negative_words.txt.")

if __name__ == "__main__":
    main()