import random
import copy
import numpy as np


class SGA_homophily(object):
    def __init__(self, attack_limit, pc, pm, population_list_num, potential_edges, homophily_decrease_score, remove_duplicate):
        self.attack_limit = attack_limit
        self.pc = pc
        self.pm = pm
        self.population_list_num = population_list_num
        self.potential_edges = potential_edges

        # decrease of node homophily
        self.homophily_decrease_score = homophily_decrease_score

        # remove the duplicated individuals or not
        self.remove_duplicate = remove_duplicate


    def initialize(self):
        pop = []
        candidate_edges = copy.deepcopy(self.potential_edges)
        for j in range(self.population_list_num):
            chromosome_id = np.random.choice(np.arange(len(candidate_edges)), size=self.attack_limit, replace=False)
            chromosome = list(candidate_edges[chromosome_id])
            pop.append(chromosome)
        return pop


    # Remove possible duplicates in the individuals (same link attacks) and between the individuals (same individual attacks)
    def solve_conflict(self, pop):
        tmp_pop = copy.deepcopy(pop)
        if self.remove_duplicate == False:
            #remove the possible duplicates in the individuals (same link attacks)
            for i in range(len(tmp_pop)):
                edges = [tmp_pop[i][j][1] for j in range(len(tmp_pop[i]))]
                remove_dup_edges = list(set(edges))
                while len(remove_dup_edges) != len(edges):
                    still_need = len(edges) - len(remove_dup_edges)
                    candidate_edges = copy.deepcopy(self.potential_edges)
                    candidate_id = np.arange(len(self.potential_edges))
                    new_add = candidate_edges[np.random.choice(candidate_id, size=still_need, replace=False)][:, 1]
                    remove_dup_edges = list(set(remove_dup_edges + list(new_add)))
                for ttt in range(len(remove_dup_edges)):
                    tmp_pop[i][ttt][1] = remove_dup_edges[ttt]
        else:
            #firstly, remove the possible duplicates in the individuals (same link attacks) first
            current_valid = []
            for i in range(len(tmp_pop)):
                edges = [tmp_pop[i][j][1] for j in range(len(tmp_pop[i]))]
                remove_dup_edges = list(set(edges))
                while len(remove_dup_edges) != len(edges):
                    still_need = len(edges) - len(remove_dup_edges)
                    candidate_edges = copy.deepcopy(self.potential_edges)
                    candidate_id = np.arange(len(self.potential_edges))
                    new_add = candidate_edges[np.random.choice(candidate_id, size=still_need, replace=False)][:, 1]
                    remove_dup_edges = list(set(remove_dup_edges + list(new_add)))
                for ttt in range(len(remove_dup_edges)):
                    tmp_pop[i][ttt][1] = remove_dup_edges[ttt]

                # then, remove the possible duplicates between the individuals (same individual attacks)
                edges = [tmp_pop[i][j][1] for j in range(len(tmp_pop[i]))]
                remove_dup_edges = sorted(edges)
                while remove_dup_edges in current_valid:
                    still_need = self.attack_limit
                    candidate_edges = copy.deepcopy(self.potential_edges)
                    candidate_id = np.arange(len(self.potential_edges))
                    new_add = candidate_edges[np.random.choice(candidate_id, size=still_need, replace=False)][:, 1]
                    remove_dup_edges = sorted(list(set(list(new_add))))
                for ttt in range(len(remove_dup_edges)):
                    tmp_pop[i][ttt][1] = remove_dup_edges[ttt]
                edges = [tmp_pop[i][j][1] for j in range(len(tmp_pop[i]))]
                current_valid.append(sorted(edges))
        return tmp_pop


    def crossover_operation(self, parents_pop):
        crossed_pop = copy.deepcopy(parents_pop)
        tag_id = list(np.arange(len(parents_pop)))
        point = int(self.attack_limit / 2)
        while (len(tag_id) > 1):
            parents_id = random.sample(tag_id, 2)
            if (random.random() > self.pc):
                parents_pop[parents_id[0]][0:point], parents_pop[parents_id[1]][0:point] = parents_pop[parents_id[1]][0:point], parents_pop[parents_id[0]][0:point]
            crossed_pop.append(parents_pop[parents_id[0]])
            crossed_pop.append(parents_pop[parents_id[1]])
            tag_id.remove(parents_id[0])
            tag_id.remove(parents_id[1])
        crossed_pop = self.solve_conflict(crossed_pop)
        return crossed_pop

    def mutation_operation(self, crossed_pop):
        mutation_pop = copy.deepcopy(crossed_pop)
        for i in range(len(mutation_pop)):
            if i < len(mutation_pop)/2:
                continue
            if (random.random() > self.pm):
                point = np.random.choice(np.arange(self.attack_limit))
                replace_edge_id = np.random.choice(np.arange(len(self.potential_edges)), size=1, replace=False)
                mutation_pop[i][point] = copy.deepcopy(self.potential_edges[replace_edge_id][0])
        mutation_pop = self.solve_conflict(mutation_pop)
        return mutation_pop


    def elite_selection(self, mutation_pop, edges_ranks_score):
        elite_pop = []
        elite_score= []
        tag_id = list(np.arange(len(mutation_pop)))
        while len(tag_id) > 1:
            id_1, id_2 = random.sample(tag_id, 2)
            tag_id.remove(id_1)
            tag_id.remove(id_2)

            # two-level sorting mechanism
            if edges_ranks_score[id_1] < edges_ranks_score[id_2]:
                elite_pop.append(copy.deepcopy(mutation_pop[id_1]))
                elite_score.append(edges_ranks_score[id_1])
            elif edges_ranks_score[id_1] > edges_ranks_score[id_2]:
                elite_pop.append(copy.deepcopy(mutation_pop[id_2]))
                elite_score.append(edges_ranks_score[id_2])
            else: #second-level sorting mechanism based on the DNP
                total_decrease_homophily1 = 0
                total_decrease_homophily2 = 0
                for i in range(len(mutation_pop[id_1])):
                    total_decrease_homophily1 += self.homophily_decrease_score[mutation_pop[id_1][i][1]]
                    total_decrease_homophily2 += self.homophily_decrease_score[mutation_pop[id_2][i][1]]
                if total_decrease_homophily2 <= total_decrease_homophily1:
                    elite_pop.append(copy.deepcopy(mutation_pop[id_1]))
                    elite_score.append(edges_ranks_score[id_1])
                else:
                    elite_pop.append(copy.deepcopy(mutation_pop[id_2]))
                    elite_score.append(edges_ranks_score[id_2])
        return elite_pop, elite_score


    def find_the_best(self, elite_pop, elite_score):
        candidate = np.where(elite_score == min(elite_score))[0]
        best_id = -1
        best_decrease_homophily = -1
        for i in range(len(candidate)):
            current_decrease_homophily = 0
            for j in range(len(elite_pop[candidate[i]])):
                current_decrease_homophily += self.homophily_decrease_score[elite_pop[candidate[i]][j][1]]
            if current_decrease_homophily > best_decrease_homophily:
                best_decrease_homophily = current_decrease_homophily
                best_id = candidate[i]
        return elite_pop[best_id], elite_score[best_id]