import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

def split_text(text, max_words=100):
    sentences = sent_tokenize(text)
    chunks = []
    chunk = []
    word_count = 0
    for sentence in sentences:
        words = sentence.split()
        if word_count + len(words) > max_words:
            chunks.append(" ".join(chunk))
            chunk = []
            word_count = 0
        chunk.extend(words)
        word_count += len(words)
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks
