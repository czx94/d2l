import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import torch

from utils import eval_embedding, vis_embedding, create_logger, Sampler
from models import SDNE
from configs import cfg
from losses import MixLoss

import networkx


def main(device):
    parser = argparse.ArgumentParser("SDNE training")
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
    model = SDNE(graph, cfg).to(device)

    L_matrix, adj_matrix = model.get_matrix()
    criterion = MixLoss(cfg, L_matrix, adj_matrix, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.WORD2VEC.LR, weight_decay=cfg.SDNE.L2)

    train(adj_matrix, model, cfg, criterion, optimizer, device)
    embedding = model.get_embedding(adj_matrix.to(device))

    eval_embedding(embedding, cfg.DATA.LABEL_PATH, logger)
    vis_embedding(embedding, cfg.DATA.LABEL_PATH, log_path)

def train(adj_matrix, model, cfg, criterion, optimizer, device):
    epochs = cfg.WORD2VEC.EPOCH
    adj_matrix = adj_matrix.to(device)

    for epoch in range(epochs):
        Y, X = model(adj_matrix)
        loss = criterion(X, Y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch:{epoch}, loss:{loss.item()}")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(device)