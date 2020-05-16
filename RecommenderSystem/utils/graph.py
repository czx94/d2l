class Graph(object):
    def __init__(self, graph_path):
        self.init_graph(graph_path)

    def init_graph(self, path):
        self.edge_count = 0
        with open(path, 'r') as f:
            my_graph = f.readlines()

        self.my_graph = dict()
        self.edges = list()
        for items in my_graph:
            key, value = items.split()

            if key not in self.my_graph:
                self.my_graph[key] = [value]
            else:
                if value in self.my_graph[key]:
                    continue
                self.my_graph[key].append(value)

            self.edges.append((key, value))
            self.edge_count += 1

            if value not in self.my_graph:
                self.my_graph[value] = list()

    def vertex_index_mapping(self):
        # generate indexes vertexs mapping
        self.v2index = {}
        self.index2v = {}
        for ind, key in enumerate(self.my_graph.keys()):
            self.v2index[key] = ind
            self.index2v[ind] = key

        return self.index2v, self.v2index

    def get_vertexs_degree(self):
        self.vertexs_degree = [0] * len(self)
        for edge in self.edges:
            self.vertexs_degree[self.v2index[edge[0]]] += 1

        return self.vertexs_degree

    def get_neighbors(self, key):
        if key in self.my_graph:
            return self.my_graph[key]
        else:
            return []

    def get_vertexs(self):
        return list(self.my_graph.keys())

    def get_edges(self):
        return self.edges

    def __len__(self):
        return len(self.my_graph)

if __name__ == '__main__':
    graph = Graph('../data/wiki/Wiki_edgelist.txt')
    for i in ['936', '1401', '16']:
        print(graph.get_neighbors(i))

    for key, value in graph.my_graph.items():
        if len(value) == 0:
            print(key)

    ind2v, v2ind = graph.vertex_index_mapping()

    degree = graph.get_vertexs_degree()


