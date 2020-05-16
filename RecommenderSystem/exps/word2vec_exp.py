import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

from models import Word2Vec
from utils import w2vDataset, cosine_similarity
from losses import NegativeSamplingLoss

import torch
from torch import optim

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def main(device):
    # configs
    data_path = "./data/tianlongbabu_fenci.txt"
    window_size = 5
    noise_ratio = 5
    min_count = 10
    batch_size = 1024
    embedding_dim = 300
    print_every = 1500
    epochs = 500
    lr = 0.001
    momentum = 0.9
    weight_decay = 3e-4

    train_dset = w2vDataset(data_path, window_size=window_size, noise_ratio=noise_ratio, min_count=min_count)
    train_dloader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=2)

    print("Data ready")
    print(f'Total words {len(train_dset)}, unique words {train_dset.vocabulary_size}.')

    # instantiating the model
    model = Word2Vec(train_dset.vocabulary_size, embedding_dim).to(device)

    # using the loss that we defined
    criterion = NegativeSamplingLoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Training set ready")

    for e in range(epochs):
        # scheduler.step()
        # get our input, target batches
        for steps, vectors in enumerate(train_dloader):
            inputs, targets, noises = vectors[0].view(-1), vectors[1].view(-1), vectors[2].view(-1, noise_ratio)
            inputs, targets, noises = inputs.to(device), targets.to(device), noises.to(device)

            # input, outpt, and noise vectors
            input_vectors = model(inputs)
            output_vectors = model.n_forward(targets)
            noise_vectors = model.n_forward(noises)

            # negative sampling loss
            loss = criterion(input_vectors, output_vectors, noise_vectors)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss stats
            if steps % print_every == 0:
                print(f"Epoch: {e+1}/{epochs}")
                print(f"Loss:{loss.item()}")  # avg batch loss at this point in training
                # print(f"Lr:{scheduler.get_lr()[0]}")

                search_words = [train_dset.vocab_to_int[x] for x in ["萧峰", "乔峰", "段誉", "虚竹"]]
                valid_examples, valid_similarities = cosine_similarity(model.embed_model, targets=search_words, device=device)

                scores, closest_idxs = valid_similarities.topk(6)

                valid_examples, closest_idxs, scores = valid_examples.to('cpu'), closest_idxs.to('cpu'), scores.to('cpu')
                for ind, valid_idx in enumerate(valid_examples):
                    closest_words = [train_dset.int_to_vocab[idx.item()] for idx in closest_idxs[ind]][1:]
                    closest_scores = [f"{closest_words[i]}: {scores[ind][i]}" for i in range(len(closest_words))]
                    print(f"{train_dset.int_to_vocab[valid_idx.item()]} | {','.join(closest_scores)}")
                print("...\n")

    # # getting embeddings from the embedding layer of our model, by name
    # embeddings = model.embed_model.weight.to('cpu').data.numpy()
    #
    # viz_words = 380
    # tsne = TSNE()
    # embed_tsne = tsne.fit_transform(embeddings[:viz_words, :])
    #
    # for idx in range(viz_words):
    #     plt.scatter(*embed_tsne[idx, :], color='steelblue')
    #     plt.annotate(train_dset.int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
    #
    # plt.gcf().savefig('vis.png')

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(device)

