import random
import numpy as np

def preprocess_nxgraph(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx

class Sampler(object):
    def __init__(self, graph, batch_size=32, negative_sampling_ratio=5, edge_sampling_mode="alias", vertex_sampling_mode="alias"):
        self.graph = graph
        self.batch_size = batch_size
        self.negative_sampling_ratio = negative_sampling_ratio
        self.edge_samping_mode = edge_sampling_mode
        self.vertex_samping_mode = vertex_sampling_mode

        self.ind2v, self.v2ind = self.graph.vertex_index_mapping()

        self.edges = self.graph.get_edges()
        self.vertexs = self.graph.get_vertexs()

        self.edge_distribution = [1/len(self.graph.edges) for _ in range(len(self.graph.edges))]
        self.edge_sampler = Alias(self.edge_distribution)

        vertex_degrees = self.graph.get_vertexs_degree()
        vertex_probs = [(degree / sum(vertex_degrees)) ** 0.75 for degree in vertex_degrees]

        self.vertex_distribution = [prob / sum(vertex_probs) for prob in vertex_probs]
        self.vertex_sampler = Alias(self.vertex_distribution)

        self.iters = len(self.edges) // batch_size + 1

    def __len__(self):
        return self.iters
    
    def fetch(self):
        if self.edge_samping_mode == "numpy":
            batch_indexes = np.random.choice(len(self.edges), size=self.batch_size, p=self.edge_distribution)
        elif self.edge_samping_mode == "alias":
            batch_indexes = self.edge_sampler.sample(self.batch_size)

        u_i = []
        u_j = []
        label = []
        for ind in batch_indexes:
            edge = self.edges[ind]
            u_i.append(self.v2ind[edge[0]])
            u_j.append(self.v2ind[edge[1]])
            label.append(1)

            for _ in range(self.negative_sampling_ratio):
                if self.vertex_samping_mode == "numpy":
                    negative_vertex = np.random.choice(len(self.ind2v), self.vertex_distribution)
                elif self.vertex_samping_mode == "alias":
                    negative_vertex = self.vertex_sampler.sample()[0]

                u_i.append(self.v2ind[edge[0]])
                u_j.append(negative_vertex)
                label.append(-1)

        return u_i, u_j, label


class Alias(object):
    def __init__(self, probs):
        self.accept, self.alias = self.create_table(probs)

    def create_table(self, probs):
        probs = [prob * len(probs) for prob in probs]
        accept, alias = [0] * len(probs), [0] * len(probs)

        high_inds, low_inds = list(), list()

        for ind, prob in enumerate(probs):
            if prob >= 1:
                high_inds.append(ind)
            else:
                low_inds.append(ind)

        while high_inds and low_inds:
            low, high = low_inds.pop(), high_inds[-1]
            accept[low] = probs[low]
            alias[low] = high
            probs[high] -= (1 - accept[low])

            if probs[high] < 1:
                high_inds.pop()
                low_inds.append(high)

        for high in high_inds:
            accept[high] = 1
        for low in low_inds:
            accept[low] = 1

        return accept, alias

    def sample(self, size=1):
        batch = []
        for _ in range(size):
            ind = random.randint(0, len(self.accept)-1)
            if random.random() < self.accept[ind]:
                batch.append(ind)
            else:
                batch.append(self.alias[ind])

        return batch


if __name__ == '__main__':
    pass
