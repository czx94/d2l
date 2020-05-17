import random
import itertools
import ray

from .sampler import Alias

class Walker(object):
    def __init__(self, graph, p=1, q=1):
        self.graph = graph
        self.p = p
        self.q = q

        if self.p != 1 or self.q != 1:
            self._generate_transition_probs()

    def deepwalk(self, seq_length, cur_node):
        seq = [cur_node]

        while len(seq) < seq_length:
            cur_node = seq[-1]
            neighbors = list(self.graph.neighbors(cur_node))
            if neighbors:
                seq.append(random.choice(neighbors))
            else:
                break

        return seq

    def node2vec(self, seq_length, cur_node):
        seq = [cur_node]
        while len(seq) < seq_length:
            cur_node = seq[-1]
            neighbors = list(self.graph.neighbors(cur_node))
            if neighbors:
                if len(seq) == 1:
                    seq.append(neighbors[self.node_alias[cur_node].sample()[0]])
                else:
                    edge = (seq[-2], seq[-1])
                    seq.append(neighbors[self.edge_alias[edge].sample()[0]])
            else:
                break

        return seq


    def meta_generator(self, num_seq, seq_length):
        nodes = list(self.graph.nodes())

        ray.init(num_cpus=4)
        generators = [self.seq_generator.remote(self, nodes, seq_length) for _ in range(num_seq)]
        seqs = ray.get(generators)

        seqs = list(itertools.chain.from_iterable(seqs))

        return seqs

    @ray.remote
    def seq_generator(self, nodes, seq_length):
        seqs = []

        random.shuffle(nodes)
        for v in nodes:
            if self.p == 1 and self.q == 1:
                seqs.append(self.deepwalk(seq_length, v))
            else:
                seqs.append(self.node2vec(seq_length, v))

        return seqs

    def _generate_transition_probs(self):
        # node alias table
        self.node_alias = dict()
        for node in self.graph.nodes():
            node_probs = [self.graph[node][nbr].get('weight', 1.0) for nbr in self.graph.neighbors(node)]
            const = sum(node_probs)
            normalized_node_prob = [prob/const for prob in node_probs]
            self.node_alias[node] = Alias(normalized_node_prob)

        # edge alias table
        self.edge_alias = dict()
        for edge in self.graph.edges():
            prev, cur = edge[0], edge[1]
            edge_probs = list()
            for node in self.graph.neighbors(cur):
                weight = self.graph[cur][node].get('weight', 1.0)
                if node == prev:
                    edge_probs.append(weight/self.p)
                elif self.graph.has_edge(node, prev):
                    edge_probs.append(weight)
                else:
                    edge_probs.append(weight/self.q)

            const = sum(edge_probs)
            normalized_edge_prob = [prob/const for prob in edge_probs]
            self.edge_alias[edge] = Alias(normalized_edge_prob)



