import os
import pandas as pd
from src.utils import load_stopwords
from src.preprocessor import DataPreprocessor
from src.text_mining import TextMiner
from src.visualization import Visualizer

DATA_DIR = "/Users/younny/Documents/work/projects/PycharmProjects/dataHandler/data"
OUTPUT_DIR = "/Users/younny/Documents/work/projects/PycharmProjects/dataHandler/output"

def main():
    # Setup
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("Loading stopwords (for Chinese)...")
    stopwords = load_stopwords()
    
    # 1. Loading & Cleaning
    print("Initializing Preprocessor...")
    preprocessor = DataPreprocessor(stopwords)
    
    print(f"Loading and processing data from {DATA_DIR}...")
    df_en, df_cn = preprocessor.load_and_clean_data(DATA_DIR)
    
    # 2. Process English Pipeline
    print(f"--- English Pipeline ({len(df_en)} docs) ---")
    if not df_en.empty:
        en_miner = TextMiner(OUTPUT_DIR, "en")
        en_miner.process(df_en)
    else:
        print("No English data found.")
        
    # 3. Process Chinese Pipeline
    print(f"--- Chinese Pipeline ({len(df_cn)} docs) ---")
    if not df_cn.empty:
        cn_miner = TextMiner(OUTPUT_DIR, "cn")
        cn_miner.process(df_cn)
    else:
        print("No Chinese data found.")
        
    # 4. Visualization
    cn_font_path = "/System/Library/Fonts/STHeiti Light.ttc"
    # Use font path if it exists, otherwise None (English handles default ok, Chinese needs it)
    if not os.path.exists(cn_font_path):
        cn_font_path = None
        print("Warning: Chinese font not found at default path. WordCloud might contain boxes.")

    visualizer = Visualizer(OUTPUT_DIR, font_path=cn_font_path)
    
    print("Generating English Word Cloud...")
    if not df_en.empty:
        visualizer.generate_wordcloud(df_en, "en")
    
    print("Generating Chinese Word Cloud...")
    if not df_cn.empty:
        visualizer.generate_wordcloud(df_cn, "cn")
        
    # 5. Heatmap
    print("Generating English Heatmap...")
    en_tfidf_path = os.path.join(OUTPUT_DIR, "en_tfidf_chi.csv")
    visualizer.generate_heatmap(en_tfidf_path, "en")
    
    print("Generating Chinese Heatmap...")
    cn_tfidf_path = os.path.join(OUTPUT_DIR, "cn_tfidf_chi.csv")
    visualizer.generate_heatmap(cn_tfidf_path, "cn")
    
    print("All tasks completed.")

if __name__ == "__main__":
    main()
