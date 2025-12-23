import os
import pandas as pd
import jieba
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from gensim.models import Phrases
from gensim.models.phrases import Phraser

class DataPreprocessor:
    def __init__(self, stopwords):
        if stopwords:
            self.stopwords = stopwords
        else:
            self.stopwords = set()
            
        # Load user dictionary if exists
        user_dict_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dict', 'custom_dict.txt')
        if os.path.exists(user_dict_path):
            jieba.load_userdict(user_dict_path)
            print(f"Loaded user dictionary from {user_dict_path}")

        # English Setup
        self.lemmatizer = WordNetLemmatizer()
        # Custom English Stopwords
        self.en_custom_stopwords = {
            'said', 'say', 'year', 'others', 'also', 'us', 'would', 'could', 'told', 'one', 'two',
            'mr', 'ms', 'mrs', 'new', 'report', 'bbc', 'like', 'time', 'people'
        }
        
    def normalize_date(self, date_str):
        try:
            return pd.to_datetime(date_str).strftime('%Y-%m-%d %H:%M:%S')
        except:
            return None

    def clean_text_english(self, text):
        # Basic cleaning: lowercase, remove non-alpha (keep spaces model training)
        import re
        text = str(text).lower()
        # Remove special chars but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    # Returns List of tokens now
    def process_english_tokens(self, text):
        try:
            nltk.data.find('tokenizers/punkt_tab')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
            nltk.data.find('corpora/omw-1.4')
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('averaged_perceptron_tagger_eng')
            
        tokens = word_tokenize(text)
        
        # Helper to map NLTK POS tags to WordNet POS tags
        from nltk.corpus import wordnet
        def get_wordnet_pos(treebank_tag):
            if treebank_tag.startswith('J'):
                return wordnet.ADJ
            elif treebank_tag.startswith('V'):
                return wordnet.VERB
            elif treebank_tag.startswith('N'):
                return wordnet.NOUN
            elif treebank_tag.startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN # Default
        
        # Tag tokens
        tagged = nltk.pos_tag(tokens)
        
        eng_stopwords = set(nltk_stopwords.words('english')).union(self.en_custom_stopwords)
        
        filtered = []
        for w, tag in tagged:
            if len(w) < 2: continue # remove single chars
            if w in eng_stopwords: continue
            
            # Lemmatize with POS
            wnet_pos = get_wordnet_pos(tag)
            lemma = self.lemmatizer.lemmatize(w, pos=wnet_pos)
            
            if lemma in eng_stopwords: continue # Check again after lemmatization ('said' -> 'say')
            if lemma == 'wa': continue # singular 'was' often lemmatized to 'wa' by mistake without context? No, 'was' -> 'be'. 'wa' is odd. 
            
            filtered.append(lemma)
        
        return filtered

    def clean_text_chinese(self, text):
        import re
        text = str(text)
        text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        return text

    def segment_chinese(self, text):
        segs = jieba.cut(text)
        filtered = [w for w in segs if w not in self.stopwords and len(w.strip()) > 0]
        return " ".join(filtered)

    def load_and_clean_data(self, data_dir):
        en_data = []
        cn_data = []
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if not file.endswith('.csv'):
                    continue
                
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue

                content_col = None
                date_col = None
                label = None

                folder_name = os.path.basename(root)
                label = folder_name
                
                if folder_name.lower() == 'bbc':
                    label = os.path.splitext(file)[0]

                if 'content' in df.columns:
                    content_col = 'content'
                elif '微博正文' in df.columns:
                    content_col = '微博正文'
                
                if not content_col:
                    continue
                
                if 'date' in df.columns:
                    date_col = 'date'
                elif '发布时间' in df.columns:
                    date_col = '发布时间'
                
                temp_df = pd.DataFrame()
                temp_df['text_raw'] = df[content_col]
                temp_df['label'] = label
                
                if date_col:
                    temp_df['date'] = df[date_col].apply(self.normalize_date)
                else:
                    temp_df['date'] = None

                temp_df.dropna(subset=['text_raw'], inplace=True)
                
                is_english = 'bbc' in file_path.lower()
                
                if is_english:
                    # Step 1: Clean & Tokenize
                    temp_df['tokens'] = temp_df['text_raw'].apply(lambda x: self.process_english_tokens(self.clean_text_english(x)))
                    # Remove empty
                    temp_df = temp_df[temp_df['tokens'].apply(len) > 0]
                    en_data.append(temp_df)
                else:
                    temp_df['text_processed'] = temp_df['text_raw'].apply(lambda x: self.segment_chinese(self.clean_text_chinese(x)))
                    temp_df = temp_df[temp_df['text_processed'].str.strip() != '']
                    cn_data.append(temp_df)
    
        df_en = pd.concat(en_data, ignore_index=True) if en_data else pd.DataFrame()
        df_cn = pd.concat(cn_data, ignore_index=True) if cn_data else pd.DataFrame()
        
        # English Bigram Processing
        if not df_en.empty:
            print("Training English Bigram Model...")
            docs = df_en['tokens'].tolist()
            # Train model
            phrases = Phrases(docs, min_count=2, threshold=2)  # Low threshold to catch many bigrams
            bigram = Phraser(phrases)
            
            # Transform
            print("Applying Bigrams...")
            df_en['text_processed'] = df_en['tokens'].apply(lambda tokens: " ".join(bigram[tokens]))
            df_en.drop(columns=['tokens'], inplace=True)
            
            # Deduplicate strictly on the final text
            df_en.drop_duplicates(subset=['text_processed'], inplace=True)
        
        if not df_cn.empty:
            df_cn.drop_duplicates(subset=['text_processed'], inplace=True)
            
        return df_en, df_cn
