from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np

class FeatureSelector:
    def __init__(self, top_k: int = 1000, min_freq: int = 5, min_tfidf: float = 0.01):
        """Initialize the selector.

        Args:
            top_k: Number of top features to keep after chi2 selection.
            min_freq: Minimum document frequency a term must have to be considered.
            min_tfidf: Minimum average TF-IDF score a feature must have to be kept.
        """
        self.top_k = top_k
        self.min_freq = min_freq
        self.min_tfidf = min_tfidf
        self.selected_features = None
        self.vectorizer = None  # Final TF‑IDF vectorizer

    def _filter_low_freq(self, count_vec: CountVectorizer, X_counts):
        """Remove terms whose document frequency is below ``min_freq``.

        Parameters
        ----------
        count_vec: CountVectorizer instance used to generate ``X_counts``.
        X_counts: sparse matrix of shape (n_samples, n_features).
        """
        # Document frequency = number of non‑zero rows per column
        doc_freq = np.asarray((X_counts > 0).sum(axis=0)).ravel()
        mask = doc_freq >= self.min_freq
        if not mask.any():
            # If nothing passes the filter, keep all to avoid empty matrix
            mask = np.ones_like(doc_freq, dtype=bool)
        # Reduce the matrix and feature names
        X_filtered = X_counts[:, mask]
        filtered_feature_names = np.array(count_vec.get_feature_names_out())[mask]
        return X_filtered, filtered_feature_names

    def chi_tfidf(self, texts, labels):
        """Perform CHI‑square feature selection followed by TF‑IDF weighting.

        This method now filters out low‑frequency terms before applying chi2,
        and optionally filters low TF-IDF weight terms after.
        """
        # 1. Count Vectorization (Bag of Words)
        count_vec = CountVectorizer()
        try:
            X_counts = count_vec.fit_transform(texts)
        except ValueError:
            # Handle empty input
            return None, None

        # 2. Filter low‑frequency terms
        X_counts, feature_names = self._filter_low_freq(count_vec, X_counts)
        print(f"[{self.__class__.__name__}] Filtered low-frequency terms: {len(feature_names)} features remaining (min_freq={self.min_freq})")

        # 3. Compute Chi2 scores
        chi2_stats, _ = chi2(X_counts, labels)

        # 4. Select top k features by Chi2
        k = min(self.top_k, len(feature_names))
        top_k_indices = np.argsort(chi2_stats)[-k:]
        self.selected_features = feature_names[top_k_indices]
        print(f"Selected {len(self.selected_features)} features via Chi‑Square (min_freq={self.min_freq}).")

        # 5. TF‑IDF on selected features
        self.vectorizer = TfidfVectorizer(vocabulary=self.selected_features)
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # 6. Filter by TF-IDF Threshold (if set)
        if self.min_tfidf is not None and self.min_tfidf > 0:
            # Calculate average TF-IDF score for each feature across all documents
            # (Note: many docs will be 0, so average might be low. 
            # Alternatively use max. But user said "TF-IDF threshold", usually implicitly means "importance".
            # Let's use max TF-IDF score for the feature to ensure it's significant in at least some docs.)
            # Or use mean of non-zero? Standard approach: sum tfidf / N?
            # Let's try: keep features where at least one document has TF-IDF > min_tfidf
            # Or average > threshold. 
            # Let's go with: Mean TF-IDF > threshold is too strict if corpus is large.
            # Max TF-IDF > threshold ensures it's relevant somewhere.
            # But the user wants to reduce noise in heatmap. Heatmap aggregates.
            # Let's use mean TF-IDF across documents. Or sum.
            # Let's use: keep features where mean TF-IDF > min_tfidf.
            # Let's start conservative.
            feature_means = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
            mask = feature_means >= self.min_tfidf
            
            if mask.sum() > 0:
                self.selected_features = self.selected_features[mask]
                tfidf_matrix = tfidf_matrix[:, mask]
                print(f"[{self.__class__.__name__}] Filtered by Min Mean TF-IDF ({self.min_tfidf}): {len(self.selected_features)} features remaining.")
            else:
                print(f"[{self.__class__.__name__}] TF-IDF filter too strict (min={self.min_tfidf}), keeping all {len(self.selected_features)}.")

        return tfidf_matrix, self.selected_features

    def get_feature_names(self):
        return self.selected_features
