import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

import argparse

from utils import Graph, eval_embedding, vis_embedding, create_logger
from models import DeepWalk
from configs import cfg


def main():
    parser = argparse.ArgumentParser("Deep walk training")
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

    graph = Graph(cfg.DATA.GRAPH_PATH)
    model = DeepWalk(graph, cfg, logger)

    model.train()
    embedding = model.get_embedding()

    eval_embedding(embedding, cfg.DATA.LABEL_PATH, logger)
    vis_embedding(embedding, cfg.DATA.LABEL_PATH, log_path)


if __name__ == '__main__':
    main()