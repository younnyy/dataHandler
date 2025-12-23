import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

class Visualizer:
    def __init__(self, output_dir, font_path=None):
        self.output_dir = output_dir
        self.font_path = font_path

    def generate_wordcloud(self, df, lang_prefix):
        if df.empty:
            print(f"[{lang_prefix}] No data for word cloud.")
            return

        text = " ".join(df['text_processed'].astype(str).tolist())
        
        # WordCloud config
        # Use font_path if provided (essential for Chinese)
        wc = WordCloud(
            font_path=self.font_path,
            width=800,
            height=600,
            background_color='white',
            max_words=200
        ).generate(text)

        # Save to file
        output_file = os.path.join(self.output_dir, f"{lang_prefix}_wordcloud.png")
        wc.to_file(output_file)
        print(f"[{lang_prefix}] Saved word cloud to {output_file}")
        
        # Optional: verify by trying to open or just print success

    def generate_heatmap(self, tfidf_csv_path, lang_prefix, top_n_features=30):
        try:
            import seaborn as sns
        except ImportError:
            print("Seaborn not installed.")
            return

        if not os.path.exists(tfidf_csv_path):
            print(f"[{lang_prefix}] TF-IDF file not found: {tfidf_csv_path}")
            return
            
        try:
            df = pd.read_csv(tfidf_csv_path)
        except Exception as e:
            print(f"[{lang_prefix}] Error reading TF-IDF file: {e}")
            return
            
        if df.empty:
            return

        # Aggregate by label (mean TF-IDF)
        # Drop date and other non-numeric cols except label
        numeric_cols = df.select_dtypes(include=[float, int]).columns
        # We need to keep label for grouping
        # But 'label' is string. 
        # So we group by label and take mean of numeric columns
        
        # Check if label exists
        if 'label' not in df.columns:
            print(f"[{lang_prefix}] Label column missing for heatmap.")
            return

        # Group by label
        heatmap_data = df.groupby('label')[numeric_cols].mean()
        
        # Select Top N features for visualization (to avoid overcrowding)
        # We pick features with the highest max importance across any label
        if heatmap_data.shape[1] > top_n_features:
            # max value per column (feature)
            max_scores = heatmap_data.max()
            top_features = max_scores.nlargest(top_n_features).index
            heatmap_data = heatmap_data[top_features]
            
        plt.figure(figsize=(12, 8))
        # Use a chinese-compatible font if provided
        if self.font_path:
            # Seaborn/Matplotlib font setting
            from matplotlib.font_manager import FontProperties
            prop = FontProperties(fname=self.font_path)
            plt.rcParams['font.family'] = prop.get_name()
            # Also sometimes need to set explicitly for sns
            sns.set(font=prop.get_name())
            
        sns.heatmap(heatmap_data, cmap="YlGnBu", annot=False)
        plt.title(f"{lang_prefix.upper()} Feature Importance Heatmap (Top {top_n_features})")
        
        # Adjust layout
        plt.tight_layout()
        
        output_file = os.path.join(self.output_dir, f"{lang_prefix}_heatmap.png")
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        print(f"[{lang_prefix}] Saved heatmap to {output_file}")
