import numpy as np

def build_vocab(text):
    vocab = sorted(set(text))
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    return vocab, char2idx, idx2char

def vectorize_string(string, char2idx):
    return np.array([char2idx[c] for c in string])
