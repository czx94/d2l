import torch
from torch import nn
import numpy as np


class LINE(nn.Module):
    def __init__(self, graph, cfg):
        super().__init__()
        self.graph = graph
        self.cfg = cfg
        self.order = cfg.LINE.ORDER

        self.num_vertexs = graph.number_of_nodes()
        self.embedding_size = self.cfg.WORD2VEC.EMBEDDING_SIZE

        self.embeddings = nn.ModuleDict()
        self.embeddings["first"] = nn.Embedding(self.num_vertexs, self.embedding_size)
        self.embeddings["second"] = nn.Embedding(self.num_vertexs, self.embedding_size)
        self.embeddings["context"] = nn.Embedding(self.num_vertexs, self.embedding_size)
        self.embedding_table = dict()
        
        self._init_model()
        self._vertex_index_mapping()

    def _vertex_index_mapping(self):
        self.v2index = {}
        self.index2v = {}
        for ind, key in enumerate(list(self.graph.nodes())):
            self.v2index[key] = ind
            self.index2v[ind] = key

    def _init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                m.weight.data.uniform_(-0.03, 0.03)

    def get_mapping(self):
        return self.v2index, self.index2v

    def get_embedding(self):
        if self.order == "first":
            weight = self.embeddings["first"].weight
        elif self.order == "second":
            weight = self.embeddings["second"].weight
        else:
            weight = torch.cat([self.embeddings["first"].weight, self.embeddings["second"].weight], dim=1)

        for v, ind in self.v2index.items():
            self.embedding_table[v] = weight[ind].detach().cpu().numpy()

        return self.embedding_table

    def forward(self, u_i, u_j, order):
        assert order in ["first", "second", "both"]
        if order == "first":
            out_i = self.embeddings[order](u_i)
            out_j = self.embeddings[order](u_j)

        elif order == "second":
            out_i = self.embeddings[order](u_i)
            out_j = self.embeddings["context"](u_j)

        else:
            out_i = self.embeddings["first"](u_i)
            out_j = self.embeddings["first"](u_j)
            out_first_order = (out_i*out_j).sum(dim=1)

            out_i = self.embeddings["second"](u_i)
            out_j = self.embeddings["context"](u_j)
            out_second_order = (out_i*out_j).sum(dim=1)

            return out_first_order + out_second_order

        out = out_i*out_j
        return out.sum(dim=1)
