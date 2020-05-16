from gensim.models import Word2Vec
from utils import Walker

class DeepWalk(object):
    def __init__(self, graph, cfg, logger):
        self.walker = Walker(graph)
        self.sentences = self.walker.meta_generator(cfg.SAMPLE.NUM_SEQUENCE, cfg.SAMPLE.SEQUENCE_LENGTH)

        self.cfg = cfg
        self.logger = logger
        self.embedding_table = dict()
        self.model = None

    def train(self):
        kwargs = dict()
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = self.cfg.WORD2VEC.MIN_COUNT
        kwargs["size"] = self.cfg.WORD2VEC.EMBEDDING_SIZE
        kwargs["sg"] = self.cfg.WORD2VEC.SG
        kwargs["hs"] = self.cfg.WORD2VEC.HS
        kwargs["workers"] = self.cfg.WORD2VEC.WORKERS
        kwargs["window"] = self.cfg.WORD2VEC.WINDOW
        kwargs["iter"] = self.cfg.WORD2VEC.ITER

        self.model = Word2Vec(**kwargs)

    def get_embedding(self):
        if not self.embedding_table:
            if self.model:
                for v in self.walker.graph.get_vertexs():
                    self.embedding_table[v] = self.model.wv[v]

        import pdb
        pdb.set_trace()

        return self.embedding_table



