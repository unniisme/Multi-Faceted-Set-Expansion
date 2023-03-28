#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 23:03:28 2019

@author: kpal
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:00:31 2019

@author: kpal
"""
import sys
# sys.path.insert(1, '/src')

import numpy as np
import scipy.sparse as sp
import json
import model.candidate_collection as cancol
import model.dataPrep as dp
import model.sim_var as sc
import model.SEISA_mod as seps
import model.structure_lib as util
from collections import defaultdict
import time
import copy



class Model:


    def initiate(self, entity_map, list_map, entity_dict, list_dict):

        self.entity_map = entity_map
        self.list_map = list_map
        self.entity_dict = entity_dict
        self.list_dict = list_dict


    def calculate(self, domain, query, n_pg = 5, n_p = 6, score_type = 1, category_score_type = 2):

        print("NPG ",n_pg)
        self.query = [query]
        self.n_pg = n_pg
        self.n_p = n_p

        self.score_type = score_type
        self.category_score_type = category_score_type
        self.domain = domain
        self.data_path =  domain + "/"

        # def getTopX(x, arr):

        
        print("loaded all dictionaries")

        peergroup_result = defaultdict(list)

        setup_time = []
        processing_time = []

        time_a = time.time()
        found_list = []
        peergroupList = []

        candidates, cl = cancol.candidate_collect(copy.deepcopy(self.query),self.entity_map, self.list_map,2,1000)

        candidate_l = cancol.list_candidate_collect(candidates, self.entity_map)

        print("PRINTING SOMETHING")
        print(len(candidate_l))
        print("Candidates_l : ",candidate_l[0:5])
        print("Candidates : ",candidates[0:5])


        entityTolist = (sp.load_npz(self.data_path+"etolist_matrix.npz")).toarray()
        print("Entity to list size : ", len(entityTolist))
        print("Entity to list[0] size : ", len(entityTolist[0]))
        # print(entityTolist[0][0])

        for i in range(10):
            print(entityTolist[0][i])

        print("loaded th matrix")
        candidate_index, candidate_l_index = [], []

        for i in candidates:
            candidate_index.append(self.entity_dict[i])
        candidate_index.sort()

        for i in candidate_l:
            candidate_l_index.append(self.list_dict[i])
        candidate_l_index.sort()

        entityTolist = cancol.sliceEtoList(entityTolist, candidate_l_index, candidate_index)

        entities = [0]*len(candidate_index)
        print("CAND : ",candidates[:5])

        seed_inds = []
        for i in candidates:
            temp = candidate_index.index(self.entity_dict[i])
            entities[temp] = i
            if i in query:
                seed_inds.append(temp)

        print("ENTS : ",entities[:10])

        group_lists = [0]*len(candidate_l_index)
        for i in candidate_l:
            group_lists[candidate_l_index.index(self.list_dict[i])] = i

        initial_q, q_index = dp.createQuery(self.query, entities)
        found_peers = []

        ###############################################


        c_score = dp.score_dict(self.data_path+"categories_scores.tsv", group_lists,1)
#        c_score = dp.score_dict(self.data_path+"filtered_categories_scores.tsv", group_lists,1)
        c_score = np.true_divide(c_score, np.amax(c_score))
#        e_score = dp.score_dict(self.data_path+"entities_scores.tsv",entities,2)
        e_score = dp.score_dict(self.data_path+"entities_scores.tsv",entities,self.score_type)
        e_score = np.true_divide(e_score, np.amax(e_score))
##        cat_size = dp.read_size(self.data_path+"categories_score.tsv", group_lists,1)
        cat_size = dp.score_dict(self.data_path+"categories_scores.tsv", group_lists,self.category_score_type)
        catsize_norm = np.true_divide(1, cat_size)


        ###############################################


        time_b = time.time() - time_a
        setup_time.append(time_b)
        time_c = time.time()
        peer_group = []
        number_entity = np.shape(entityTolist)[0]
        S = np.zeros((number_entity, number_entity))

        # entities => list of E entities, according to the index 
        # group_lists => list of F facets, according to index

        print("Entity to list size : ", len(entityTolist))
        print("Entity to list[0] size : ", len(entityTolist[0]))

        # Matrix E x F ==> 1 if E belongs in F, 0 otherwise
        print("Entitytolist : ", entityTolist)

        # FACET SCORE --> List of (F) scores
        print("c_score size : ", len(c_score))
        print("c_score : ", c_score)

        # ENTITY SCORE --> List of (E) scores 
        print("e_score size : ", len(e_score))
        print("e_score : ", e_score)

        print("Group_lists : ", len(group_lists))
        print("Group_lists : ", group_lists[:5])

        print("Query :" , query)
        print("Initial_q :" , initial_q)



        #-----------------------------------------------------------
        

        # print("Entity to list size : ", len(entityTolist))
        # print("Entitytolist : ", entityTolist)
        # print(len(entityTolist))


        # for i in range(self.n_pg):
        #     print("done creating s matrix")
        #     D = dp.createdismatrix(S)
        #     print("done creating d matrix")
        #     peer_group, c_index, group_score = seps.findGroup_Listconst_discomp_score(initial_q, S, self.n_p, entityTolist, found_peers, D, self.list_map, group_lists, catsize_norm, found_list)

        #     group = []
        #     for j in peer_group:
        #         if ((j in found_peers)!= True):
        #             found_peers.append(j)
        #         group.append(entities[j])

        #     common_list = dp.updateMatrix(group, self.query, entityTolist, entities)
        #     peer_lists = []
        #     for j in common_list:
        #         peer_lists.append(group_lists[j])

        #     g = util.PeerGroup()
        #     g.initiate(i, group, peer_lists, group_score[0], group_score[1], group_score[2], group_lists[c_index])
        #     peergroupList.append(g)

        alpha = 0.3
        beta = 0.3
        gamma = 0.4

        relMat = np.around(sc.createSimMatrix(entityTolist, S, c_score, e_score, []), decimals = 2)
        

        # entity_dict
        entityTolistC = np.zeros((len(entityTolist), len(entityTolist[0])))
        entityTolistI = []

        print("SEED INDS : ", seed_inds)
        print(entities[seed_inds[0]])
        for i in seed_inds:
            for k in range(len(c_score)):
                for j in range(len(e_score)):
                    if(entityTolist[j][k] == 1):
                        entityTolistC[j][k] += alpha * relMat[i][j] 

        

        cnt = 0
        for i in range(len(e_score)):
            if(entityTolistC[i][seed_inds[0]] > 0):
                cnt += 1 

        print("CNT", cnt)
        print("ENTITY TO LIST : ", entityTolistC)
        print("RELMAT : ",relMat)
        peerGroups = []

    
        x = 2
        groupScores = []
        cnt = 0

        # -------------------------------- ALGORITHM 2 --------------------
        
        # STEP 1-2
        # STEP FOR FINDING HIGH x SCORES FOR EACH FACET
        for j in range(len(c_score)):        
            inds = set()
            inds_1 = []
            for i in range(x):
                mini = -100000000
                for k in range(len(e_score)):
                    if(k in inds_1):
                        continue
                    if(entityTolistC[k][j] > mini):
                        ind = k 
                        mini = entityTolistC[k][j]
                inds.add(tuple([-1 * mini, ind]))
                inds_1.append(ind)

            peerGroups.append(inds)

        print("AFTER ALPHA ADDITION")
        print(peerGroups[:10])

        # STEP 3-5
        for k in range(len(c_score)):
            for j in range(len(e_score)):
                for i in peerGroups[k]:
                    entityTolistC[j][k] += beta * (relMat[j][i[1]])


        # STEP 6-7
        peerGroups = []
        for j in range(len(c_score)):        
            inds = set()
            inds_1 = []
            for i in range(x):
                mini = -1
                for k in range(len(e_score)):
                    if(k in inds_1):
                        continue
                    if(entityTolistC[k][j] > mini):
                        ind = k 
                        mini = entityTolistC[k][j]
                inds.add(tuple([-1 * mini, ind]))
                inds_1.append(ind)

            peerGroups.append(inds)

        print("AFTER BETA ADDITION")
        print(peerGroups[:10])

        # STEP 8-12
        for i in range(len(c_score)):
            for k in peerGroups[i]:
                for j in range(len(c_score)):
                    if(i == j):
                        continue 
                    for k1 in peerGroups[j]:
                        entityTolistC[k[1]][i] -= gamma * relMat[k1[1]][k[1]]

        peerGroups = []
        for j in range(len(c_score)):        
            inds = set()
            inds_1 = []
            for i in range(x):
                mini = -1
                for k in range(len(e_score)):
                    if(k in inds_1):
                        continue
                    if(entityTolistC[k][j] > mini):
                        ind = k 
                        mini = entityTolistC[k][j]
                inds.add(tuple([-1 * mini, ind]))
                inds_1.append(ind)

            peerGroups.append(inds)


        print("AFTER GAMMA ADDITION")
        print(peerGroups[:10])

# /(x*(len(peerGroups)-1))
        # Calculating Group Score
        for i in range(len(c_score)):
            sm = 0
            for j in peerGroups[i]:
                # sm += entityTolistC[j[1]][i]
                sm += (-1*j[0])
            groupScores.append(sm)

        print("GROUP SCORES :")
        # print(groupScores)
        #  To store top k groups
        topGroups = set()
        tgInd = set()
        for _ in range(n_pg):
            mex = -1000000000
            for i in range(len(c_score)):
                if(i in tgInd):
                    continue
                if(groupScores[i] > mex):
                    ind = i
                    mex = groupScores[i]
            topGroups.add(tuple([-1*mex, ind]))
            tgInd.add(ind)

        print(topGroups)
        print("PRINTING TOP K GROUPS ITER 1 ", len(topGroups))
        for i in topGroups:
            print(group_lists[i[1]], -1*i[0])
            # print(peerGroups[i[1]])
            for j in peerGroups[i[1]]:
                print(entities[j[1]], -1*j[0])



        # DEBUGGING FOR CHECKING peerGroups

        # cnt = 0
        # for i in peerGroups:
        #     # print(i)
        #     # if(i[0])
        #     # print("Starting Peer", len(i))
        #     if (next(iter(i))[0] == 0):
        #         print(group_lists[cnt])
        #     # for j in i:
        #     cnt += 1
            #     if(j[0] == 0):

            #         break
                # print(entities[j[1]])
            # print("Ending Peer")




        print("done calculating peer group")
        # peergroup_result[self.query[0]] = [peergroupList]
        peergroup_result[self.query[0]] = []

        time_d = time.time() - time_c
        processing_time.append(time_d)

        count = 0
        for i in setup_time:
            count += i
        print("avg setup time: ", count / len(self.query))

        count = 0
        for i in processing_time:
            count += i
        print("avg setup time: ", count / len(self.query))

        return peergroup_result
    
       #---------------------------------------------------------
       # IBM
        # type+wikicat_Publicly_traded_companies_of_the_United_States
        # type+wikicat_Electronics_companies_of_the_United_States
        # type+wikicat_Companies_in_the_Dow_Jones_Industrial_Average
        # type+wordnet_company_108058098
        # type+wikicat_Software_companies_of_the_United_States
        # done calculating peer group
        # avg setup time:  0.2951045036315918
        # avg setup time:  29.955456256866455
        # -------------------------------------------------------
