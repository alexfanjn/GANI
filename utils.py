from deeprobust.graph.utils import accuracy
from deeprobust.graph.defense import GCN
import torch
import numpy as np


def asr(time_, features, adj, modified_features, modified_adj, labels, idx_test, idx_train=None, idx_val=None, retrain=False):
    gcn = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
              nhid=16, dropout=0.5, with_relu=False, with_bias=True, device='cpu').to('cpu')
    gcn.load_state_dict(torch.load('ori_model/ori_sgc'+time_+'.pkl'))
    output = gcn.predict(features,adj)
    acc1 = np.float(accuracy(output[idx_test],labels[idx_test]))
    if retrain:
        gcn.fit(modified_features, modified_adj, labels, idx_train, idx_val, patience=30)
    modified_output = gcn.predict(modified_features, modified_adj)
    acc2 = np.float(accuracy(modified_output[idx_test],labels[idx_test]))

    print('The accuracy before the attacks:', acc1)
    print('The accuracy after the attacks:', acc2)
    return acc1, acc2



def evasion_test(node_injection, time_, features, adj, idx_train, idx_val, idx_test):
    modified_features = node_injection.modified_features
    modified_adj = node_injection.modified_adj
    n_added_labels = node_injection.n_added_labels
    acc1, acc2 = asr(time_, features, adj, modified_features, modified_adj, n_added_labels, idx_test, idx_train, idx_val)
    return acc1, acc2


def poisoning_test(node_injection, time_, features, adj, idx_train, idx_val, idx_test):
    modified_features = node_injection.modified_features
    modified_adj = node_injection.modified_adj
    n_added_labels = node_injection.n_added_labels
    acc1, acc2 = asr(time_, features, adj, modified_features, modified_adj, n_added_labels, idx_test,
                     idx_train, idx_val, retrain=True)
    return acc1, acc2