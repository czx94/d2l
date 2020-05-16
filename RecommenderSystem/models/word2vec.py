import torch
from torch import nn

class Word2Vec(nn.Module):
    def __init__(self, num_vocab, num_emb):
        ''''''
        super().__init__()

        self.num_vocab = num_vocab
        self.num_emb = num_emb

        self.embed_model = nn.Embedding(self.num_vocab, self.num_emb)
        self.embed_model.weight.data.uniform_(-0.5 / self.num_emb, 0.5 / self.num_emb)

        self.neighbor_embed = nn.Embedding(self.num_vocab, self.num_emb)
        self.neighbor_embed.weight.data.uniform_(-0, 0)


    def forward(self, words):
        vectors = self.embed_model(words)
        return vectors

    def n_forward(self, words):
        vectors = self.neighbor_embed(words)
        return vectors

