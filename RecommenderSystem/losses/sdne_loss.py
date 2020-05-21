import torch
from torch import nn

class MixLoss(nn.Module):
    def __init__(self, cfg, L_matrix, adj_matrix, device):
        super().__init__()
        self.alpha = cfg.SDNE.ALPHA
        self.beta = cfg.SDNE.BETA
        self.L_matrix = L_matrix.to(device)
        self.adj_matrix = adj_matrix.to(device)

        self.matrix_beta = torch.ones_like(adj_matrix)
        self.matrix_beta[self.matrix_beta==0] = self.beta
        self.matrix_beta = self.matrix_beta.to(device)

    def forward(self, X, Y):
        second_order_loss = ((X - self.adj_matrix)*self.matrix_beta).pow(2).sum(dim=0).mean()

        trace = Y.t().mm(self.L_matrix).mm(Y).trace()
        first_order_loss = 2*self.alpha*trace

        return first_order_loss + second_order_loss


if __name__ == "__main__":
    L_matrix = torch.rand((256, 256))
    adj_matrix = torch.rand((256, 256))
    X = torch.rand((256, 256))
    Y = torch.rand((256, 64))
    criterion = MixLoss(1e-4, 5, L_matrix, adj_matrix)
    loss = criterion(X, Y)

    import pdb
    pdb.set_trace()
