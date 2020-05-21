import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import scipy.sparse as sp

class SDNE(nn.Module):
    def __init__(self, graph, cfg):
        super().__init__()
        self.graph = graph
        self.cfg = cfg
        self.hidden_layers = cfg.SDNE.LAYERS
        self.node_number = self.graph.number_of_nodes()

        self.encoder = nn.ModuleList()
        for i in range(len(self.hidden_layers)):
            if i == 0:
                self.encoder.append(nn.Sequential(
                    nn.Linear(self.node_number, self.hidden_layers[i]),
                    nn.ReLU(),
                ))
            else:
                self.encoder.append(nn.Sequential(
                    nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]),
                    nn.ReLU(),
                ))

        self.decoder = nn.ModuleList()
        for i in reversed(range(len(self.hidden_layers))):
            if i == 0:
                self.decoder.append(nn.Sequential(
                    nn.Linear(self.hidden_layers[i], self.node_number),
                    nn.ReLU(),
                ))
            else:
                self.decoder.append(nn.Sequential(
                    nn.Linear(self.hidden_layers[i], self.hidden_layers[i-1]),
                    nn.ReLU(),
                ))

        self._vertex_index_mapping()
        self._init_matrix()

    def _init_matrix(self):
        self.adj_matrix = torch.zeros((self.node_number, self.node_number))
        for edge in self.graph.edges():
            i, j = self.v2index[edge[0]], self.v2index[edge[1]]
            self.adj_matrix[i][j] += self.graph[edge[0]][edge[1]].get('weight', 1.0)

        self.adj_sym_matrix = self.adj_matrix + self.adj_matrix.t()

        self.D_matrix = torch.diag(self.adj_sym_matrix.sum(dim=1))

        self.L_matrix = self.D_matrix - self.adj_sym_matrix


    def _vertex_index_mapping(self):
        self.v2index = {}
        self.index2v = {}
        for ind, key in enumerate(list(self.graph.nodes())):
            self.v2index[key] = ind
            self.index2v[ind] = key

    def forward(self, X):
        for layer in self.encoder:
            X = layer(X)
        Y = X

        for layer in self.decoder:
            X = layer(X)
        X_ = X

        return Y, X_

    def get_embedding(self, X):
        for layer in self.encoder:
            X = layer(X)
        embeddings = X.detach().cpu().numpy()

        self.embeddings = dict()
        for ind, embedding in enumerate(embeddings):
            self.embeddings[self.index2v[ind]] = embedding

        return self.embeddings

    def get_matrix(self):
        return self.L_matrix, self.adj_matrix

