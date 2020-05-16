from torch import nn

class KLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inner_product, sign):
        return -(inner_product*sign).sigmoid().log().mean()