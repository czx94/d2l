import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

import argparse
import networkx

from utils import eval_embedding, vis_embedding, create_logger
from models import Struct2Vec
from configs import cfg


def main():
    parser = argparse.ArgumentParser("Struct2vec training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger, log_path = create_logger(cfg)
    logger.info(cfg)

    graph = networkx.read_edgelist(cfg.DATA.GRAPH_PATH, create_using=networkx.DiGraph(), nodetype=None, data=[('weight', int)])
    model = Struct2Vec(graph, cfg, logger)

    model.train()
    embedding = model.get_embedding()

    eval_embedding(embedding, cfg.DATA.LABEL_PATH, logger)
    vis_embedding(embedding, cfg.DATA.LABEL_PATH, log_path)


if __name__ == '__main__':
    main()