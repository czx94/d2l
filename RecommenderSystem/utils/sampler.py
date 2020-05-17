import random
import numpy as np

class Sampler(object):
    def __init__(self, graph, v2ind, batch_size=32, negative_sampling_ratio=5, edge_sampling_mode="alias", vertex_sampling_mode="alias"):
        self.graph = graph
        self.v2ind = v2ind
        self.batch_size = batch_size
        self.negative_sampling_ratio = negative_sampling_ratio
        self.edge_samping_mode = edge_sampling_mode
        self.vertex_samping_mode = vertex_sampling_mode

        self.edges = list(self.graph.edges())
        self.vertexs = list(self.graph.nodes())

        edge_weight = [self.graph[edge[0]][edge[1]].get('weight', 1.0) for edge in self.edges]
        const = sum(edge_weight)
        self.edge_distribution = [weight * self.graph.number_of_edges()/const for weight in edge_weight]
        self.edge_sampler = Alias(self.edge_distribution)

        vertex_degrees = np.zeros(len(self.vertexs))
        for edge in self.graph.edges():
            vertex_degrees[self.v2ind[edge[0]]] += self.graph[edge[0]][edge[1]].get('weight', 1.0)
        const = sum(vertex_degrees)
        vertex_probs = [(degree / const) ** 0.75 for degree in vertex_degrees]

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
