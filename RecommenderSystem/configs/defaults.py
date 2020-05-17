import os

from yacs.config import CfgNode as CN

_C = CN()

_C.EXP = CN()
_C.EXP.NAME = "name"
_C.EXP.LOG_PATH = "./logs"

_C.DATA = CN()
_C.DATA.GRAPH_PATH = "./data/wiki/Wiki_edgelist.txt"
_C.DATA.LABEL_PATH = "./data/wiki/wiki_labels.txt"
_C.DATA.CATGORY_PATH = "./data/wiki/Wiki_category.txt"

_C.SAMPLE = CN()
_C.SAMPLE.SEQUENCE_LENGTH = 10
_C.SAMPLE.NUM_SEQUENCE = 80
_C.SAMPLE.WORKERS = 2
_C.SAMPLE.BATCHSIZE = 512
_C.SAMPLE.P = 0.25
_C.SAMPLE.Q = 4

_C.WORD2VEC = CN()
_C.WORD2VEC.MIN_COUNT = 5
_C.WORD2VEC.EMBEDDING_SIZE = 300
_C.WORD2VEC.SG = 1
_C.WORD2VEC.HS = 1
_C.WORD2VEC.WORKERS = 4
_C.WORD2VEC.WINDOW = 5
_C.WORD2VEC.ITER = 5
_C.WORD2VEC.EPOCH = 50
_C.WORD2VEC.LR = 0.001

_C.LINE = CN()
_C.LINE.ORDER = "second"

