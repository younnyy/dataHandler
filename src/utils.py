import os
import glob

original_stopwords_path = "/Users/younny/Documents/work/projects/PycharmProjects/dataHandler/stopwords"
def load_stopwords(stopwords_dir=original_stopwords_path):
    """
    Load stopwords from all text files in the specified directory.
    """
    stopwords = set()
    if not os.path.exists(stopwords_dir):
        print(f"Warning: Stopwords directory '{stopwords_dir}' does not exist.")
        return stopwords

    file_paths = glob.glob(os.path.join(stopwords_dir, "*.txt"))
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        stopwords.add(word)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    print(f"Loaded {len(stopwords)} unique stopwords from {len(file_paths)} files.")
    return stopwords
