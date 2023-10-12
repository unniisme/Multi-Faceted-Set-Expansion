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
import candidate_collection as cancol
import dataPrep as dp
import sim_var as sc
import SEISA_mod as seps
import structure_lib as util
from collections import defaultdict
import time
import copy

import logging
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)13s() ] %(message)s"
logging.basicConfig(format=FORMAT, filename='test.log', encoding='utf-8', level=logging.DEBUG)


class Model:


    def initiate(self, entity_map, list_map, entity_dict, list_dict):

        self.entity_map = entity_map
        self.list_map = list_map
        self.entity_dict = entity_dict
        self.list_dict = list_dict


    def calculate(self, domain, query, n_pg = 5, n_p = 6, score_type = 1, category_score_type = 2):

        logging.info("NPG %s",n_pg)
        self.query = [query]
        self.n_pg = n_pg
        self.n_p = n_p

        self.score_type = score_type
        self.category_score_type = category_score_type
        self.domain = domain
        self.data_path =  domain + "/"

        # def getTopX(x, arr):

        
        logging.info("loaded all dictionaries")

        peergroup_result = defaultdict(list)

        setup_time = []
        processing_time = []

        time_a = time.time()
        found_list = []
        peergroupList = []

        candidates, _ = cancol.candidate_collect(copy.deepcopy(self.query),self.entity_map, self.list_map,2,1000)

        candidate_l = cancol.list_candidate_collect(candidates, self.entity_map)

        logging.debug("PRINTING SOMETHING")
        logging.debug("length of Candidates_l : %s", len(candidate_l))
        logging.debug("candidates_l : %s",candidate_l[0:5])
        logging.debug("candidates : %s",candidates[0:5])

        # enitityToList is a mapping  between entities and lists
        entityTolist = (sp.load_npz(self.data_path+"etolist_matrix.npz")).toarray()
        logging.debug("Entity to list size : %s", len(entityTolist))
        logging.debug("Entity to list[0] size : %s", len(entityTolist[0]))

        # for i in range(100):
            # logging.debug(entityTolist[0][i])
        logging.debug("Entity to List mapping:\n%s", entityTolist[:10, :10])
        logging.info("loaded th matrix")
        candidate_index, candidate_l_index = [], []

        for i in candidates:
            candidate_index.append(self.entity_dict[i])
        candidate_index.sort()

        for i in candidate_l:
            candidate_l_index.append(self.list_dict[i])
        candidate_l_index.sort()

        entityTolist = cancol.sliceEtoList(entityTolist, candidate_l_index, candidate_index)

        entities = [0]*len(candidate_index)

        seed_inds = []
        for i in candidates:
            temp = candidate_index.index(self.entity_dict[i])
            entities[temp] = i
            if i in query:
                seed_inds.append(temp)

        logging.debug("Entities : %s",entities[:10])

        group_lists = [0]*len(candidate_l_index)
        for i in candidate_l:
            group_lists[candidate_l_index.index(self.list_dict[i])] = i

        logging.debug("Group Lists : %s",group_lists[:10])

        initial_q, _ = dp.createQuery(self.query, entities) # BUG: WHYYYYYYY
        # found_peers = []

        ###############################################


        c_score = dp.score_dict(self.data_path+"categories_scores.tsv", group_lists,1)
#        c_score = dp.score_dict(self.data_path+"filtered_categories_scores.tsv", group_lists,1)
        c_score = np.true_divide(c_score, np.amax(c_score))
#        e_score = dp.score_dict(self.data_path+"entities_scores.tsv",entities,2)
        e_score = dp.score_dict(self.data_path+"entities_scores.tsv",entities,self.score_type)
        e_score = np.true_divide(e_score, np.amax(e_score))
##        cat_size = dp.read_size(self.data_path+"categories_score.tsv", group_lists,1)
        # cat_size = dp.score_dict(self.data_path+"categories_scores.tsv", group_lists,self.category_score_type)
        # catsize_norm = np.true_divide(1, cat_size)


        ###############################################


        time_b = time.time() - time_a
        setup_time.append(time_b)
        time_c = time.time()
        peer_group = []
        number_entity = np.shape(entityTolist)[0]
        S = np.zeros((number_entity, number_entity))

        # entities => list of E entities, according to the index 
        # group_lists => list of F facets, according to index

        logging.info("Entity to list size : %s", len(entityTolist))
        logging.info("Entity to list[0] size : %s", len(entityTolist[0]))

        # Matrix E x F ==> 1 if E belongs in F, 0 otherwise
        logging.info("Entitytolist :\n%s", entityTolist)

        # FACET SCORE --> List of (F) scores
        logging.info("c_score size : %s", len(c_score))
        logging.info("c_score :\n%s", c_score)

        # ENTITY SCORE --> List of (E) scores 
        logging.info("e_score size : %s", len(e_score))
        logging.info("e_score : \n%s", e_score)

        logging.info("Group_lists : %s", len(group_lists))
        logging.info("Group_lists : \n%s", group_lists[:5])

        logging.info("Query : %s" , query)
        logging.info("Initial_q : \n%s" , initial_q)



        #-----------------------------------------------------------
        

        # print("Entity to list size : ", len(entityTolist))
        # print("Entitytolist : ", entityTolist)
        # print(len(entityTolist))


        # for i in range(self.n_pg):
        #     print("done creating s matrix")
        #     D = dp.createdismatrix(S)
        #     print("done creating d matrix")
        # #     peer_group, c_index, group_score = seps
        # logging.debug(initial_q).findGroup_Listconst_discomp_score(initial_q, S, self.n_p, entityTolist, found_peers, D, self.list_map, group_lists, catsize_norm, found_list)

        #     group = []
        #     for j in peer_group:
        # #         if ((j in found_peers)!= True
        # logging.debug(initial_q)):
        # #             found_peers.append(j
        # logging.debug(initial_q))
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

        logging.debug("SEED INDS : %s", seed_inds)
        logging.debug(entities[seed_inds[0]])
        for i in seed_inds:
            for k in range(len(c_score)):
                for j in range(len(e_score)):
                    if(entityTolist[j][k] == 1):
                        entityTolistC[j][k] += alpha * relMat[i][j] 

        

        cnt = 0
        for i in range(len(e_score)):
            if(entityTolistC[i][seed_inds[0]] > 0):
                cnt += 1 

        logging.debug("CNT %s", cnt)
        logging.debug("ENTITY TO LIST :  %s", entityTolistC)
        logging.debug("RELMAT :  %s",relMat)
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

        logging.debug("AFTER ALPHA ADDITION")
        logging.debug(peerGroups[:10])

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

        logging.debug("AFTER BETA ADDITION")
        logging.debug(peerGroups[:10])

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


        logging.debug("AFTER GAMMA ADDITION")
        logging.debug(peerGroups[:10])

# /(x*(len(peerGroups)-1))
        # Calculating Group Score
        for i in range(len(c_score)):
            sm = 0
            for j in peerGroups[i]:
                # sm += entityTolistC[j[1]][i]
                sm += (-1*j[0])
            groupScores.append(sm)

        logging.info("GROUP SCORES :")
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

        logging.info(topGroups)
        logging.info("PRINTING TOP K GROUPS ITER 1 (len: %s)", len(topGroups))
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




        logging.info("done calculating peer group")
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
        print("avg processing time: ", count / len(self.query))

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
