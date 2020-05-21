import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

import re
import random
from collections import Counter

class w2vDataset(Dataset):
    def __init__(self, text_path, window_size, noise_ratio, min_count=10):
        self.window_size = window_size
        self.noise_ratio = noise_ratio
        self.min_count = min_count

        self.preprocess(text_path)

        # self.generate_pairs()


    def preprocess(self, path):
        with open(path) as f:
            text = f.read()

        # get list of words
        words = preprocess(text)

        # create lookup table for words
        self.vocab_to_int, self.int_to_vocab = create_lookup_tables(words)
        int_words = [self.vocab_to_int[word] for word in words]

        # filter words by frequency
        threshold = 1e-5
        word_counts = Counter(int_words)

        total_count = len(int_words)
        freqs = {word: count / total_count for word, count in word_counts.items()}
        p_drop = {word: 1 - np.sqrt(threshold / freqs[word]) for word in word_counts}

        train_words = [word for word in int_words if random.random() < (1 - p_drop[word])]

        # filter by min_count
        freq_words = []
        for word in train_words:
            if word_counts[word] >= self.min_count:
                freq_words.append(word)

        word_freqs = np.array(sorted(freqs.values(), reverse=True))
        unigram_dist = word_freqs / word_freqs.sum()
        noise_dist = torch.from_numpy(unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75)))

        self.words = freq_words
        self.size = len(self.words)

        self.noise_dist =  noise_dist
        self.num_vocab = len(self.vocab_to_int)

    def generate_pairs(self):
        # sample noise vectors
        if self.noise_dist is None:
            # Sample words uniformly
            noise_dist = torch.ones(self.num_vocab)
        else:
            noise_dist = self.noise_dist

        self.pairs = []
        for idx in range(len(self.words)):
            x = self.words[idx]

            R = np.random.randint(1, self.window_size + 1)
            start = idx - R if (idx - R) > 0 else 0
            stop = start + self.window_size if (start + self.window_size) < self.size else self.size - 1
            neighbors = self.words[start:idx] + self.words[idx + 1:stop + 1]

            for n in neighbors:
                noise_words = torch.multinomial(noise_dist, self.noise_ratio, replacement=True).view(-1, self.noise_ratio)
                self.pairs.append((x, n, noise_words))

        print("Training pairs generated")


    @property
    def vocabulary_size(self):
        return self.num_vocab

    def __len__(self):
        return self.size#len(self.pairs)

    def __getitem__(self, idx):
        # positive pair
        x = self.words[idx]

        R = np.random.randint(1, self.window_size + 1)
        start = idx - R if (idx - R) > 0 else 0
        stop = start + self.window_size if (start + self.window_size) < self.size else self.size - 1
        y = random.choice(self.words[start:idx] + self.words[idx + 1:stop + 1])

        # sample noise vectors
        if self.noise_dist is None:
            # Sample words uniformly
            noise_dist = torch.ones(self.num_vocab)
        else:
            noise_dist = self.noise_dist

        # Sample words from our noise distribution
        noise_words = torch.multinomial(noise_dist, self.noise_ratio, replacement=True).view(-1, self.noise_ratio)

        return x, y, noise_words
        # return self.pairs[idx]



def preprocess(text):
    # Replace punctuation with tokens so we can use them in our model
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()

    # Remove all words with  5 or fewer occurences
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > 5]

    return trimmed_words

def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: Two dictionaries, vocab_to_int, int_to_vocab
    """
    word_counts = Counter(words)
    # sorting the words from most to least frequent in text occurrence
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    # create int_to_vocab dictionaries
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

if __name__ == '__main__':
    data_path = "./data/tianlongbabu_fenci.txt"
    window_size = 5
    noise_ratio = 5
    batch_size = 1

    train_dset = w2vDataset(data_path, window_size=window_size, noise_ratio=noise_ratio)
    train_dloader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=2)

    iterator = iter(train_dloader)
    vectors = next(iterator)

    import pdb
    pdb.set_trace()
