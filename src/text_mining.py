import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2

class TextMiner:
    def __init__(self, output_dir, lang_prefix):
        self.output_dir = output_dir
        self.prefix = lang_prefix

    def process(self, df):
        if df.empty:
            print(f"[{self.prefix}] No data to process.")
            return

        texts = df['text_processed'].tolist()
        labels = df['label'].tolist()
        dates = df['date'].tolist()

        # 1. Save Processed Data
        processed_path = f"{self.output_dir}/{self.prefix}_processed.csv"
        df[['date', 'label', 'text_processed']].to_csv(processed_path, index=False, encoding='utf-8-sig')
        print(f"[{self.prefix}] Saved processed data to {processed_path}")

        # 2. Build Dictionary & BoW
        # Use CountVectorizer to build vocab and count matrix
        count_vec = CountVectorizer()
        try:
            X_counts = count_vec.fit_transform(texts)
        except ValueError:
            print(f"[{self.prefix}] Error in vectorization (empty vocab?).")
            return

        # Save Dictionary
        vocab = count_vec.vocabulary_  # dict {word: index}
        # Sort by index
        sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])
        dict_path = f"{self.output_dir}/{self.prefix}_dictionary.txt"
        with open(dict_path, 'w', encoding='utf-8') as f:
            for word, idx in sorted_vocab:
                f.write(f"{word} {idx}\n")
        print(f"[{self.prefix}] Saved dictionary to {dict_path}")

        # Save BoW
        # Format: date, label, vector_string (or dense columns? Sparse is better for text but CSV doesn't support sparse nicely).
        # User requirement: "词袋向量文件". Usually this means a matrix.
        # Given potential size, we should probably output a sparse rep or indices, but let's try dense for now if size permits, 
        # or a custom format like "idx:count idx:count".
        # Let's check requirements. "处理成字典和词袋向量".
        # To be safe and compatible with standard models (OLDA often takes "doc_id time label word:count..."), 
        # let's write a format that is easy to parse.
        # For this experiment, let's output a CSV where the vector is a string "idx:count ..." or similar? 
        # Or standard Document-Term Matrix in CSV (might be huge).
        # Let's stick to a format that keeps 'date' and 'label'.
        # Since specific CHI is needed later, maybe just standard CSV with columns for top-k words is what they mean by "Feature output"?
        # But for BoW before selection, vocab is huge. outputting full matrix to CSV is bad idea.
        # Im going to output a format: date, label, bow (string "word_id:count ...")
        
        bow_lines = []
        X_counts_csc = X_counts.tocsc()
        X_counts_csr = X_counts.tocsr()
        
        for i in range(len(texts)):
             # Get non-zero elements
            row = X_counts_csr[i]
            # row.indices, row.data
            bow_str = " ".join([f"{idx}:{count}" for idx, count in zip(row.indices, row.data)])
            bow_lines.append(bow_str)
            
        bow_df = pd.DataFrame({
            'date': dates,
            'label': labels,
            'bow_vector': bow_lines
        })
        bow_path = f"{self.output_dir}/{self.prefix}_bow.csv"
        bow_df.to_csv(bow_path, index=False, encoding='utf-8-sig')
        print(f"[{self.prefix}] Saved BoW to {bow_path}")

        # 3. CHI-TFIDF Feature Selection
        from src.feature_selection import FeatureSelector
        
        # Initialize selector with filters
        # min_freq=5: Remove words appearing in <5 docs
        # min_tfidf=0.01: Lowered threshold to keep more features while filtering absolute noise
        selector = FeatureSelector(top_k=1000, min_freq=5, min_tfidf=0.01)
        
        tfidf_matrix, selected_features = selector.chi_tfidf(texts, labels)
        
        if tfidf_matrix is None or selected_features is None:
             print(f"[{self.prefix}] Feature selection resulted in empty set.")
             return

        # Construct TFIDF output
        dense_tfidf = tfidf_matrix.toarray()
        
        # Create DF
        tfidf_df = pd.DataFrame(dense_tfidf, columns=selected_features)
        tfidf_df.insert(0, 'label', labels)
        tfidf_df.insert(0, 'date', dates)
        
        tfidf_path = f"{self.output_dir}/{self.prefix}_tfidf_chi.csv"
        tfidf_df.to_csv(tfidf_path, index=False, encoding='utf-8-sig')
        print(f"[{self.prefix}] Saved CHI-TFIDF matrix to {tfidf_path}")
