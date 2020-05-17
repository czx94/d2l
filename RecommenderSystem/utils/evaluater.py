from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np
import matplotlib.pyplot as plt

import os
import torch
import random


def cosine_similarity(embedding, targets=[], device='cpu'):
    """ Returns the cosine similarity of validation words with words in the embedding matrix.
        Here, embedding should be a PyTorch embedding module.
    """

    # Here we're calculating the cosine similarity between some random words and
    # our embedding vectors. With the similarities, we can look at what words are
    # close to our random words.

    # sim = (a . b) / |a||b|

    embed_vectors = embedding.weight

    # magnitude of embedding vectors, |b|
    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)

    # pick N words from our ranges (0,window) and (1000,1000+window). lower id implies more frequent
    valid_examples = torch.LongTensor(targets).to(device)

    valid_vectors = embedding(valid_examples)
    valid_vectors_magnitudes = valid_vectors.pow(2).sum(dim=1).rsqrt().view(-1, 1)
    similarities = torch.mm(valid_vectors, embed_vectors.t()) * valid_vectors_magnitudes / magnitudes

    return valid_examples, similarities

def eval_embedding(embedding, label_path, logger):
    X, Y = read_vertices_label(label_path)

    tr_frac = 0.8
    logger.info("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(logger, embedding=embedding, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def vis_embedding(embedding, label_path, log_path):
    X, Y = read_vertices_label(label_path)

    emb_list = []
    for k in X:
        emb_list.append(embedding[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    vertice_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(vertice_pos[idx, 0], vertice_pos[idx, 1], label=c)
    plt.legend()
    plt.savefig(os.path.join(log_path, "clusters.png"))


def read_vertices_label(path):
    vertices, labels = [], []
    with open(path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        vertice, label = line.strip().split(' ')
        vertices.append(vertice)
        labels.append([label])

    return vertices, labels


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return np.asarray(all_labels)


class Classifier(object):
    def __init__(self, logger, embedding, clf):
        self.logger = logger
        self.embedding = embedding
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)

        X_train = [self.embedding[x] for x in X]
        Y = self.binarizer.transform(Y)

        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        results['acc'] = accuracy_score(Y,Y_)
        self.logger.info('-------------------')
        self.logger.info(results)
        return results

    def predict(self, X, top_k_list):
        X_ = np.asarray([self.embedding[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        state = np.random.get_state()

        training_size = int(train_precent * len(X))
        np.random.seed(seed)
        shuffle_indices = np.random.permutation(np.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        np.random.set_state(state)
        return self.evaluate(X_test, Y_test)






