import random
import itertools

import ray

class Walker(object):
    def __init__(self, graph, p=1, q=1):
        self.graph = graph
        self.p = p
        self.q = q

    def deepwalk(self, seq_length, cur_node):
        seq = [cur_node]

        while len(seq) < seq_length:
            cur_node = seq[-1]
            neighbors = self.graph.get_neighbors(cur_node)
            if neighbors:
                seq.append(random.choice(neighbors))
            else:
                break

        return seq

    def node2vec(self, seq_length, cur_node):
        pass

    def meta_generator(self, num_seq, seq_length):
        vertexs = self.graph.get_vertexs()

        ray.init(num_cpus=4)
        generators = [self.seq_generator.remote(self, vertexs, seq_length) for _ in range(num_seq)]
        seqs = ray.get(generators)

        seqs = list(itertools.chain.from_iterable(seqs))

        return seqs

    @ray.remote
    def seq_generator(self, vertexs, seq_length):
        seqs = []

        random.shuffle(vertexs)
        for v in vertexs:
            if self.p == 1 and self.q == 1:
                seqs.append(self.deepwalk(seq_length, v))
            else:
                seqs.append(self.node2vec(seq_length, v))

        return seqs



