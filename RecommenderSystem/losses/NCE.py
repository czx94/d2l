import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_vectors, output_vectors, noise_vectors):
        # losses = []
        #
        # pos_score = torch.mul(input_vectors, output_vectors)
        # pos_score = torch.sum(pos_score, dim=1)
        # pos_score = F.logsigmoid(pos_score).mean()
        #
        # losses.append(pos_score)
        #
        # for i in range(noise_vectors.size(1)):
        #     neg_score = torch.mul(input_vectors, noise_vectors[:,i,:])
        #     neg_score = torch.sum(neg_score, dim=1)
        #     neg_score = F.logsigmoid(-1 * neg_score).mean()
        #
        #     losses.append(neg_score)
        #
        # return -1 * sum(losses)

        batch_size, embed_size = input_vectors.shape

        # Input vectors should be a batch of column vectors
        input_vectors = input_vectors.view(batch_size, embed_size, 1)

        # Output vectors should be a batch of row vectors
        output_vectors = output_vectors.view(batch_size, 1, embed_size)

        # bmm = batch matrix multiplication
        # correct log-sigmoid loss
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        out_loss = out_loss.squeeze()

        # incorrect log-sigmoid loss
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)  # sum the losses over the sample of noise vectors

        # negate and sum correct and noisy log-sigmoid losses
        # return average batch loss
        return -(out_loss + noise_loss).mean()

if __name__ == '__main__':
    criterion = NegativeSamplingLoss()
    inputs = torch.ones((64, 300))
    outputs = torch.ones((64, 300))
    noises = torch.zeros((64, 5, 300))

    loss = criterion(inputs, outputs, noises)