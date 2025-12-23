import pandas as pd
from collections import Counter
import os

OUTPUT_DIR = "output"

def get_top_k_words(csv_path, k=10, lang=""):
    if not os.path.exists(csv_path):
        print(f"[{lang}] File not found: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[{lang}] Error reading file: {e}")
        return

    if 'text_processed' not in df.columns:
        print(f"[{lang}] 'text_processed' column missing.")
        return

    # Join all text and split
    # Handle NaN values just in case
    all_text = " ".join(df['text_processed'].dropna().astype(str).tolist())
    words = all_text.split()
    
    counter = Counter(words)
    top_k = counter.most_common(k)
    
    print(f"\n--- {lang} Top {k} Words ---")
    print(f"{'Word':<20} | {'Count':<5}")
    print("-" * 30)
    for word, count in top_k:
        print(f"{word:<20} | {count:<5}")

def main():
    en_path = os.path.join(OUTPUT_DIR, "en_processed.csv")
    cn_path = os.path.join(OUTPUT_DIR, "cn_processed.csv")
    
    get_top_k_words(en_path, 10, "English")
    get_top_k_words(cn_path, 10, "Chinese")

if __name__ == "__main__":
    main()
