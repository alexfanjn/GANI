import torch
import numpy as np
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from copy import deepcopy
from time import time
from utils import *
import os
from deeprobust.graph.data import Dataset
from node_injection import NI





if __name__ == '__main__':


    # cora, citeseer, cora_ml, pubmed
    dataset_ = 'cora'
    dataset_split_seed = 2021

    # path to save the final adversarial adjacency metrix
    file1 = 'data/after_adj.npz'
    file2 = 'data/after_x.npz'
    # obtain the LCC

    root = os.getcwd() + os.sep + 'dataset'


    if not os.path.exists(root):
        os.mkdir(root)

    data = Dataset(root=root, setting='nettack', name=dataset_, seed=dataset_split_seed)


    adj, features, labels = data.adj, data.features, data.labels
    # relevant parameters
    # candidate ratio
    homophily_ratio = 0.5
    # with (False) or without (True) ground truth label
    use_predict_labels = False

    # number of injected nodes
    n_added = 10
    # or n_added = int(adj.shape[0]*0.01) * 5

    # parameters of genetic algorithm
    population_size = 40
    global_iterations = 100
    # crossover rate and mutation rate
    pc = 0.5
    pm = 0.7

    # mapping matrix to handle continuous feature
    zeroone_features = deepcopy(features)
    zeroone_features[zeroone_features > 0] = 1

    # train, val, and test set
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    # Setup normal model
    gcn = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
              nhid=16, dropout=0.5, with_relu=True, with_bias=True, device='cpu').to('cpu')
    gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)

    # Setup Surrogate model
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                    nhid=16, dropout=0.5, with_relu=False, with_bias=True, device='cpu').to('cpu')
    surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)

    # save the clean sgc model
    time_ = str(int(time()))
    torch.save(surrogate.state_dict(), 'ori_model/ori_sgc' + time_ + '.pkl')

    # the predicted label obtained from SGC model, it works when 'use_predict_labels == True'
    pred_labels = np.array(surrogate.predict(features, adj).argmax(1))

    node_injection = NI(features, adj, labels, surrogate, gcn, use_predict_labels, homophily_ratio, pred_labels, zeroone_features)

    # calculate the original homophily information of each node
    score = []
    decrease_score = []
    new_score = []
    if use_predict_labels == True:
        tmp = pred_labels
    else:
        tmp = labels
    for i in range(len(tmp)):
        current_neighbors = np.where(adj.A[i] == 1)[0]
        current_neighbors_length = len(current_neighbors)
        current_neighbors_same_label_length = len(np.where(tmp[current_neighbors] == tmp[i])[0])
        score.append(current_neighbors_same_label_length / current_neighbors_length)
        decrease_score.append(current_neighbors_same_label_length / (
                    current_neighbors_length * current_neighbors_length + current_neighbors_length))
    decrease_score = np.array(decrease_score)
    index = np.argsort(-np.array(decrease_score))
    node_injection.homophily_index = index
    node_injection.homophily_score = score
    node_injection.homophily_decrease_score = decrease_score

    mean_degree = int(adj.sum() / adj.shape[0])

    print('Begin injecting', n_added, 'nodes using GANI attacks')


    node_injection.train(n_added, time_, mean_degree, features, adj, file1, file2, use_predict_labels, global_iterations, pc, pm, population_size, idx_train, idx_val, idx_test)

    # The adversarial adjacency matrix and feature matrix are saved in '\data\xxx'.