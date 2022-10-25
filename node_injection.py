import torch
import numpy as np
import pandas as pd
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import accuracy
from deeprobust.graph import utils
from copy import deepcopy
import scipy.sparse as sp
from utils import *
import os
import random
from deeprobust.graph.data import Dataset

from ga_homophily import SGA_homophily
from utils import poisoning_test


class NI:
    def __init__(self, features, adj, labels, surrogate, model, use_predict_labels, homophily_ratio, pred_labels, zeroone_features):
        self.surrogate = surrogate
        self.model = model
        self.features = deepcopy(features)
        self.modified_features = deepcopy(features)
        self.adj = deepcopy(adj)
        self.modified_adj = deepcopy(adj)
        self.labels = labels
        self.classes = list(set(labels))
        self.injected_nodes_classes = []
        self.injected_nodes_origin = []
        self.n_class = len(set(labels))
        self.features_dim = features.shape[1]
        self.nnodes = adj.shape[0]
        self.ori_nodes = []
        self.mean_degree = int(adj.sum() / self.nnodes) + 1
        # feature-attack budget
        self.average_features_nums = np.diff(features.tocsr().indptr).mean()
        # Construct the statistical information of features
        self.get_sorted_features(use_predict_labels, zeroone_features)
        self.major_features_nums = int(self.average_features_nums)
        self.major_features_candidates = self.features_id[:, :self.major_features_nums]
        # degree sampling related
        sample_array1 = np.zeros(self.nnodes, dtype='int')
        for i in range(self.nnodes):
            current_degree1 = int(adj[i].sum())
            # maximal link-attack budget: max(current degree, 2 * mean degree)
            mean_d = self.mean_degree * 2
            if current_degree1 >= mean_d:
                sample_array1[i] = mean_d
            else:
                sample_array1[i] = current_degree1
        self.sample_array1 = sample_array1
        self.ratio = homophily_ratio
        self.pred_labels = pred_labels
        self.homophily_index = []
        self.homophily_decrease_score = []
        self.homophily_score = []

    # Statistical information of features of each class
    def get_sorted_features(self, use_predict_labels, zeroone_features):
        MAX_N = 100
        features_id = []
        feature_avg = []
        for label in self.classes:
            if use_predict_labels:
                label_features = self.features[self.pred_labels == label].toarray()
                zeroone_label_features = zeroone_features[self.pred_labels == label].toarray()
            else:
                label_features = self.features[self.labels == label].toarray()
                zeroone_label_features = zeroone_features[self.labels == label].toarray()
            count = zeroone_label_features.sum(0)
            real_count = label_features.sum(0)
            count[count == 0] = 1
            current_avg = real_count / count
            df = pd.DataFrame(columns=['count', 'features_id'],
                              data={'count': count[count.nonzero()[0]], 'features_id': count.nonzero()[0]})
            df = df.sort_values('count', ascending=False)
            df.name = 'Label ' + str(label)
            features_id.append(df['features_id'].values[:MAX_N])
            feature_avg.append(current_avg)
        self.features_id = np.array(features_id)
        self.feature_avg = np.array(feature_avg)

    # Contruct adversarial features based on the assigned label
    # Currently, n_added is only work on n_added = 1, which is the sequential generation
    def make_statistic_features(self, added_node, n_added, n_added_labels=None, rand=True):
        if rand:
            labels = np.random.choice(self.classes, n_added)
        else:
            labels = n_added_labels
        self.injected_nodes_classes[added_node] = labels[0]
        n_added_features = np.zeros((n_added, self.features_dim))
        for i in range(n_added):
            n_added_features[0][self.major_features_candidates[labels[0]]] = self.feature_avg[0][
                self.major_features_candidates[labels[0]]]
        return n_added_features

    # weight matrices of SGC model
    def get_linearized_weight(self):
        W = self.surrogate.gc1.weight @ self.surrogate.gc2.weight
        return W.detach().cpu().numpy()

    # Obtain the initial single-link attack candidate
    # Remove the corresponding links with the same label
    def get_potential_edges(self, use_predict_labels, added_node_label):
        new_candadite = []
        if use_predict_labels:
            [new_candadite.append(i) for i in deepcopy(self.homophily_index) if
             not i in np.where(self.pred_labels == added_node_label)[0]]
        else:
            [new_candadite.append(i) for i in deepcopy(self.homophily_index) if
             not i in np.where(self.labels == added_node_label)[0]]
        new_candadite = np.array(new_candadite)
        size = int(len(new_candadite))
        return np.column_stack((np.tile(self.nnodes, size), new_candadite[:size]))

    # Calculate the attack score of corresponding adversarial links
    def get_edges_scores_ranks(self, potential_edges, modified_adj, modified_features, use_predict_labels, idx_test):
        modified_adj = modified_adj.tolil()
        fw = modified_features @ self.W
        edges_scores = []
        labels = []
        labels.extend(self.labels)
        labels.extend(self.injected_nodes_classes)
        labels = np.array(labels)
        self.n_added_labels = labels
        for current_array in potential_edges:
            current_array = np.array(current_array).flatten()
            ori_node = current_array[0]
            temp = 1
            n_added_node_list = []
            while temp < len(current_array):
                n_added_node_list.append(current_array[temp])
                temp += 2
            n_added_node_list = sorted(list(set(n_added_node_list)))

            for kkk in range(len(n_added_node_list)):
                modified_adj[n_added_node_list[kkk], ori_node] = 1
                modified_adj[ori_node, n_added_node_list[kkk]] = 1
            modified_adj_norm = utils.normalize_adj(modified_adj)

            logits = (modified_adj_norm @ modified_adj_norm @ fw)
            predict_class = logits.argmax(1)
            if use_predict_labels:
                surrogate_losses = -np.sum([predict_class[i] != self.pred_labels[i] for i in idx_test])
            else:
                surrogate_losses = -np.sum([predict_class[i] != labels[i] for i in idx_test])
            edges_scores.append(surrogate_losses)

            for kkk in range(len(n_added_node_list)):
                modified_adj[n_added_node_list[kkk], ori_node] = 0
                modified_adj[ori_node, n_added_node_list[kkk]] = 0
        return edges_scores, potential_edges

    # Construct the adversarial adj matrix based on the adversarial links
    def get_modified_adj_by_edges_ranks(self, modified_adj, scores, edges, verbose=True):
        edges = np.array(edges)
        scores = np.array(scores)
        temp = 1
        modified_adj = modified_adj.tolil()
        ori_node = edges[0]
        while temp < len(edges):
            n_added_node = edges[temp]
            self.ori_nodes.extend([n_added_node])
            modified_adj[n_added_node, ori_node] = 1
            modified_adj[ori_node, n_added_node] = 1
            temp += 2
        if verbose:
            print("Edge perturbation: {} , loss: {}".format(edges, scores))
        return modified_adj.tocsr()

    # Parameter initialization
    def train_init(self, linearized=True):
        if linearized:
            self.W = self.get_linearized_weight()

    # Main attack function
    def attack_edges(self, n_added, mean_degree, use_predict_labels, global_iterations, pc, pm, population_size, idx_test, verbose=True):
        # generate the link-attack budget
        selected_degree_distrubution = random.sample(list(self.sample_array1), n_added)
        # sequential injection attacks
        for added_node in range(n_added):  # 0 ~ n_added - 1
            if selected_degree_distrubution[added_node] > 2 * mean_degree:
                selected_degree_distrubution[added_node] = int(2 * mean_degree)
            if verbose:
                print("\n\n##### Attack injected node with ID {} #####".format(
                    added_node))
                print("##### Performing {} edges #####".format(selected_degree_distrubution[added_node]))
            # randomly assign a label to current new node
            added_node_label = np.random.choice(self.classes, 1)[0]
            self.injected_nodes_origin.append(added_node_label)
            self.injected_nodes_classes.append(added_node_label)
            # generate the features to current new node
            added_node_feature = self.make_statistic_features(added_node, 1, [added_node_label], False)
            # reshape the matrices
            modified_adj = utils.reshape_mx(deepcopy(self.modified_adj),
                                            shape=(self.nnodes + 1, self.nnodes + 1)).tolil()
            modified_features = sp.vstack((self.modified_features, added_node_feature)).tocsr()
            # construct the single-link attack list
            first_potential_edges = self.get_potential_edges(use_predict_labels, added_node_label)


            edges_ranks_score, edges_ranks = self.get_edges_scores_ranks(first_potential_edges, modified_adj,
                                                                         modified_features, use_predict_labels, idx_test)



            edges_ranks_zip = zip(edges_ranks_score, self.homophily_decrease_score[edges_ranks[:, 1]], edges_ranks)
            edges_ranks_zip = sorted(edges_ranks_zip,
                                     key=lambda edges_ranks_zip: (edges_ranks_zip[0], -edges_ranks_zip[1]))
            edges_ranks_scores_list = list(zip(*edges_ranks_zip))
            edges_ranks = np.array(edges_ranks_scores_list[2])
            fianl_potential_edges = edges_ranks[0: int(len(edges_ranks) * self.ratio)]
            best_single_link_loss = edges_ranks_scores_list[0][0]


            # begin genetic algorithm to find the optimal neighbors
            if selected_degree_distrubution[added_node] != 1:
                sga = SGA_homophily(selected_degree_distrubution[added_node], pc, pm, population_size,
                                 fianl_potential_edges, self.homophily_decrease_score, remove_duplicate=True)


                parents_pop = sga.initialize()

                current_iters = 0
                elite_edge_score = 0
                iters = global_iterations
                while current_iters < iters:
                    # if current_iters % (iters/10) == 0:
                    #     print('\n\ncurrent iters: ', current_iters)
                    #     print('current best loss: ', elite_edge_score)
                    crossed_pop = sga.crossover_operation(parents_pop)
                    mutation_pop = sga.mutation_operation(crossed_pop)
                    edges_ranks_score, edges_ranks = self.get_edges_scores_ranks(mutation_pop, modified_adj,
                                                                                 modified_features,
                                                                                 use_predict_labels, idx_test)
                    elite_pop, elite_score = sga.elite_selection(mutation_pop, edges_ranks_score)
                    elite_edge, elite_edge_score = sga.find_the_best(elite_pop, elite_score)
                    parents_pop = elite_pop
                    current_iters += 1
            else:  # if only need to attack 1 edge, we can directly use the output of single-link attacks and do not need to employ GA
                elite_edge = fianl_potential_edges[0]
                elite_edge_score = best_single_link_loss
            elite_edge = np.array(elite_edge).flatten()
            # obtain the final adj
            modified_adj = self.get_modified_adj_by_edges_ranks(modified_adj, elite_edge_score, elite_edge)
            # update homophily related information
            tag_id = 1
            if use_predict_labels == True:
                tmp = self.pred_labels
            else:
                tmp = self.labels
            tmp = tmp.tolist()
            tmp.extend(self.injected_nodes_origin)
            tmp = np.array(tmp)
            while tag_id < len(elite_edge):
                current_neighbors = np.where(modified_adj.A[elite_edge[tag_id]] == 1)[0]
                current_neighbors = list(current_neighbors)
                current_neighbors.remove(self.nnodes)
                current_neighbors_len = len(np.where(modified_adj.A[elite_edge[tag_id]] == 1)[0]) - 1
                current_samelabel_neighbors_len = len(
                    np.where(tmp[current_neighbors] == tmp[elite_edge[tag_id]])[0])
                self.homophily_score[elite_edge[tag_id]] = current_samelabel_neighbors_len / (
                            current_neighbors_len + 1)
                self.homophily_decrease_score[elite_edge[tag_id]] = current_samelabel_neighbors_len / (
                            current_neighbors_len * current_neighbors_len + current_neighbors_len)
                tag_id += 2
            self.homophily_index = np.argsort(-np.array(self.homophily_decrease_score))

            # inject the current node to the original adj and f matrices
            self.modified_features = modified_features
            self.modified_adj = modified_adj
            self.nnodes = self.nnodes + 1
            if verbose:
                print("##### Injected node with ID {} is attacked finished #####".format(added_node))

    # Begin the attacks and final evaluation
    def train(self, n_added, time_, mean_degree, features, adj, file1, file2, use_predict_labels, global_iterations, pc, pm, population_size, idx_train, idx_val, idx_test, verbose=True):
        self.train_init()
        self.attack_edges(n_added, mean_degree, use_predict_labels, global_iterations, pc, pm, population_size, idx_test, verbose)
        print('\nFinish attacks\n')

        poisoning_test(self, time_, features, adj, idx_train, idx_val, idx_test)
        modified_features = self.modified_features
        modified_adj = self.modified_adj
        # save corresponding matrices
        sp.save_npz(file1, modified_adj)
        sp.save_npz(file2, modified_features)



