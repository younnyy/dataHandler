import sys
import os
sys.path.append(os.getcwd())

from src.utils import load_stopwords
from src.preprocessor import DataPreprocessor

def test_debug():
    print("--- Debugging Chinese Stopwords ---")
    stopwords = load_stopwords()
    
    test_words = ["微博", "视频", "黑鹰", "背鳍", "说", "更", "月", "日"]
    for w in test_words:
        if w in stopwords:
            print(f"[PASS] '{w}' is in stopwords.")
        else:
            print(f"[FAIL] '{w}' is NOT in stopwords!")

    print("\n--- Debugging English Cleaning ---")
    dp = DataPreprocessor(stopwords)
    
    text = "He said that they say nice things about new reports."
    tokens = dp.process_english_tokens(text)
    print(f"Input: {text}")
    print(f"Output Tokens: {tokens}")
    
    if "said" in tokens:
        print("[FAIL] 'said' was NOT filtered.")
    else:
        print("[PASS] 'said' was filtered.")
        
    if "say" in tokens:
        print("[FAIL] 'say' was NOT filtered.")
    else:
        print("[PASS] 'say' was filtered.")
        
    # Check lemmatizer specific behavior
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    print(f"Lemmatize 'said' (default): {lemmatizer.lemmatize('said')}")
    print(f"Lemmatize 'said' (verb): {lemmatizer.lemmatize('said', pos='v')}")

if __name__ == "__main__":
    test_debug()
