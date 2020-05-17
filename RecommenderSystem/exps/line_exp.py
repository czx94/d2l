import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import torch

from utils import eval_embedding, vis_embedding, create_logger, Sampler
from models import LINE
from configs import cfg
from losses import KLLoss

import networkx


def main(device):
    parser = argparse.ArgumentParser("LINE training")
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
    model = LINE(graph, cfg).to(device)
    v2ind, ind2v = model.get_mapping()
    sampler = Sampler(graph, v2ind, batch_size=cfg.SAMPLE.BATCHSIZE)

    criterion = KLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.WORD2VEC.LR)

    train(sampler, model, cfg, criterion, optimizer, device)
    embedding = model.get_embedding()

    eval_embedding(embedding, cfg.DATA.LABEL_PATH, logger)
    vis_embedding(embedding, cfg.DATA.LABEL_PATH, log_path)

def train(sampler, model, cfg, criterion, optimizer, device):
    order = cfg.LINE.ORDER
    epochs = cfg.WORD2VEC.EPOCH

    for epoch in range(epochs):
        for iter in range(len(sampler)):
            u_i, u_j, label = sampler.fetch()

            u_i, u_j, label = torch.LongTensor(u_i).to(device), torch.LongTensor(u_j).to(device), torch.Tensor(label).to(device)
            outputs = model(u_i, u_j, order)
            loss = criterion(label, outputs)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch:{epoch}, loss:{loss.item()}")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(device)