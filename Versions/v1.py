"""
 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
 
        candidates, cl = cancol.candidate_collect(copy.deepcopy(self.query),self.entity_map, self.list_map,2,2000)
 
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
 
 #        [[0.174 0.174 0.174 0.174 0.174 0.    0.    0.    0.    0.    0.    0.
 #  0.    0.    0.    0.    0.   ]
 # [0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
 #  0.    0.    0.    0.    0.   ]
 # [0.    0.171 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
 #  0.    0.171 0.171 0.    0.   ]
 # [0.165 0.165 0.    0.    0.165 0.    0.    0.    0.    0.    0.    0.
 #  0.    0.    0.    0.    0.   ]
 # [0.    0.162 0.    0.    0.    0.162 0.    0.    0.    0.162 0.    0.
 #  0.    0.    0.    0.    0.   ]
 # [0.174 0.    0.    0.    0.    0.    0.174 0.174 0.    0.    0.174 0.174
 #  0.    0.    0.    0.    0.   ]
 # [0.    0.    0.    0.    0.    0.    0.183 0.183 0.    0.    0.183 0.
 #  0.    0.    0.    0.    0.   ]
 # [0.171 0.    0.    0.    0.    0.    0.171 0.171 0.    0.    0.    0.
 #  0.    0.    0.    0.    0.171]
 # [0.    0.    0.    0.    0.    0.    0.    0.129 0.    0.    0.    0.
 #  0.129 0.    0.    0.129 0.   ]]

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


        # TOY SET, create own entity to list

        # UNCOMMENT BELOW LINES FOR TOY
        # ----------------------
        # entityTolist = []

        # for j in range(len(candidates)):
        #     temp = []
        #     for i in range(len(candidate_l)):
        #         if(entities[j] in self.list_map[group_lists[i]]):
        #             temp.append(1)
        #         else:
        #             temp.append(0)
        #     entityTolist.append(temp)
        #------------------------------


        # COMMENT BELOW LINES FOR TOY
        # ----------------------------
        entityTolist = cancol.sliceEtoList(entityTolist, candidate_l_index, candidate_index)
        # ----------------------------
        # print("ENTTOLIST : ", entityTolist)


         # [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] - JFK
         # [0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0] - ABE
         # [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0] - FDG
         # [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] - DRS
         # [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] - GRR
         # [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0] - HSB
         # [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0] - MPL
         # [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1] - OTT
         # [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0] - KIR

 
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

        # for i in range(len(e_score)):
        #     print(entities[i])

        # for i in range(len(c_score)):
        #     print(group_lists[i])

        print()

        # 
 
        alpha = 0.3
        beta = 0.3
        gamma = 0.4
 
        relMat = np.around(sc.createSimMatrix(entityTolist, S, c_score, e_score, []), decimals = 2)
        

        
        # RELMAT :  [[1.  0.59 0.6  0.57 0.56 0.42 0.4  0.35 0.34]
                #   [0.59 1.   0.57 0.55 0.54 0.61 0.61 0.58 0.43]
                #   [0.6  0.57 1.   0.72 0.68 0.47 0.46 0.41 0.4 ]
                #   [0.57 0.55 0.72 1.   0.65 0.48 0.46 0.41 0.4 ]
                #   [0.56 0.54 0.68 0.65 1.   0.48 0.47 0.42 0.4 ]
                #   [0.42 0.61 0.47 0.48 0.48 1.   0.77 0.76 0.54]
                #   [0.4  0.61 0.46 0.46 0.47 0.77 1.   0.8  0.56]
                #   [0.35 0.58 0.41 0.41 0.42 0.76 0.8  1.   0.63]
                #   [0.34 0.43 0.4  0.4  0.4  0.54 0.56 0.63 1.  ]]


        # entity_dict

        entityTolistC = np.zeros((len(entityTolist), len(entityTolist[0])))
        time_init = time.time()

        alphaScores = np.zeros(len(e_score))
 
        print("SEED INDS : ", seed_inds)
        print(entities[seed_inds[0]])
        for i in seed_inds:
            for k in range(len(c_score)):
                for j in range(len(e_score)):
                    if(j in seed_inds):
                        continue

                    if(entityTolist[j][k] == 1):
                        # print(entities[j], group_lists[k])
                        entityTolistC[j][k] += ((alpha * relMat[i][j])/len(seed_inds))
                        alphaScores[j] = entityTolistC[j][k]
 
 
 
        # cnt = 0
        # for i in range(len(e_score)):
        #     if(entityTolistC[i][seed_inds[0]] > 0):
        #         cnt += 1 
 
        # print("CNT", cnt)
        # print("ENTITY TO LIST : ", entityTolistC)
        # print("RELMAT : ",relMat)
        peerGroups = [0]*len(c_score)
 
 
        x = 2
        # groupScores = []
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
                if(mini != 0):
                    inds.add(tuple([mini, ind]))
                    inds_1.append(ind)
 
            peerGroups[j] = inds
 
        print("AFTER ALPHA ADDITION")
        # print(peerGroups[:10])
 
        print("Time:", time.time() - time_c)
        time_c = time.time()
 
        # Calculating Group Score
        groupScores = []
        print("Top K groups")
        for i in range(len(c_score)):
            sm = 0
            for j in peerGroups[i]:
                # sm += entityTolistC[j[1]][i]
                sm += j[0] / len(peerGroups[i])
            groupScores.append(sm)
 
        topGroups = []
        tgInd = set()

        mex = -1000000000
        for i in range(len(c_score)):
            if(groupScores[i] > mex):
                ind = i
                mex = groupScores[i]
        topGroups.append([mex, ind])
        tgInd.add(ind)

        for i in range(len(c_score)):
            if(i in tgInd):
                continue
            if(groupScores[i] >= mex * 0.95):
                topGroups.append([groupScores[i], i])
                tgInd.add(i)
 
        topGroups = sorted(topGroups, reverse=True)
        print(topGroups[:n_pg])
        for i in topGroups[:n_pg]:
            print(group_lists[i[1]], i[0])
            # print(peerGroups[i[1]])
            for j in peerGroups[i[1]]:
                print(entities[j[1]], j[0])
            print()

 
        # STEP 3-5
        # betaScores = np.zeros((len(entityTolist), len(entityTolist[0])))
        betaScores = np.zeros(len(e_score))
        counts = np.zeros((len(entityTolist), len(entityTolist[0])))

        # xC2 = (x * (x - 1))/2
        for k in topGroups:
            for j in peerGroups[k[1]]:
                # if(j in seed_inds):
                #         continue
                for i in peerGroups[k[1]]:
                    if(i[1] == j[1]):
                        continue
                    
                    # if(entityTolist[j][k[1]] == 1):
                    # if(k[1] == 1147):
                        # print(entities[j[1]], entities[i[1]], group_lists[k[1]])
                        # print("ADDING")
                        # print(relMat[j[1]][i[1]])
                    # if(k[1] == 1077):
                        # print(entities[j[1]], entities[i[1]], group_lists[k[1]])
                        # print("ADDING")
                        # print(relMat[j[1]][i[1]])
                    # betaScores[j][k[1]] += ((beta * (relMat[j][i[1]])))
                        # counts[j][k[1]] += 1
                    # if(k[1] == 1147):
                    #     print("adding ", entities[j[1]], entities[i[1]], relMat[j[1]][i[1]])
                    temp = len(peerGroups[k[1]])
                    groupScores[k[1]] += ((beta*(relMat[j[1]][i[1]]))) / (( (temp) * (temp - 1) )) 
                    # groupScores[k[1]] /= temp

                    if(temp != 1):
                        betaScores[i[1]] += beta * relMat[j[1]][i[1]] /  (temp - 1)

        print("AFTER BETA ADDITION")
        # print(peerGroups[:10])
        # print("PRINTING TOP K GROUPS ITER 1 ", len(topGroups))
        # for i in topGroups[:10]:
        #     print(group_lists[i[1]], i[0])
        #     # print(peerGroups[i[1]])
        #     for j in peerGroups[i[1]]:
        #         print(entities[j[1]], j[0])
        #     print()
 
        print("Time:", time.time() - time_c)
        time_c = time.time()
 
        # Calculating Group Score
        # groupScores = []
        print("Top K groups")
        # for i in topGroups:
        #     sm = 0
        #     for j in peerGroups[i[1]]:
        #         # sm += entityTolistC[j[1]][i]
        #         sm += j[0]
        #     groupScores[i[1]] = sm
 
        topGroups_new = []
        tgInd = set()

        mex = -1000000000
        for i in topGroups:
            if(groupScores[i[1]] > mex):
                ind = i[1]
                mex = groupScores[i[1]]
        topGroups_new.append([mex, ind])
        tgInd.add(ind)

        for i in topGroups:
            if(i[1] in tgInd):
                continue
            if(groupScores[i[1]] >= 0.8 * mex):
                topGroups_new.append([groupScores[i[1]], i[1]])
                tgInd.add(i[1])
 
        topGroups_new = sorted(topGroups_new, reverse=True)


        topGroups = copy.deepcopy(topGroups_new)
        for i in topGroups[:n_pg]:
            print(group_lists[i[1]], i[0])
            # print(peerGroups[i[1]])
            for j in peerGroups[i[1]]:
                print(entities[j[1]], j[0])
            print()
 
       
        topKGroups = []
        topKGroups.append(topGroups[0])
        done_set = set()
        done_set.add(topKGroups[0][1])

        print("TopGroups : ", topKGroups)
        

        for i in range(n_pg - 1):
            top_score = -100000
            ind_to_add = -1            
            for j in topGroups:
                if j[1] in done_set:
                    continue
                temp = 0
                count = 0
                for ent in peerGroups[j[1]]:
                    for k in topKGroups:
                        for ents in peerGroups[k[1]]:
                            temp -= gamma * (relMat[ent[1]][ents[1]])
                            count += 1

                if(count == 0):
                    print("Count = 0 for", group_lists[j[1]])
                    continue
                if((j[0] + ((temp)/count)) > top_score):
                    top_score = j[0] + ((temp)/count)
                    ind = j[1]

            print("Chose next facet", group_lists[ind])
            print("Score Obtained : ", top_score)
            topKGroups.append([top_score, ind])
            done_set.add(ind)

            for ii in topGroups:
                if ii[1] == ind:
                    ii[0] = top_score

        print("GROUP SCORES :")
       
        print("PRINTING TOP K GROUPS ITER 1 ", len(topKGroups))
        for i in topKGroups:
            print(group_lists[i[1]], i[0])
            # print(peerGroups[i[1]])
            for j in peerGroups[i[1]]:
                print(entities[j[1]], j[0])
            print()


        

 
 ########################################################################################
 # Iterative Algorithm  
 ########################################################################################
 
        v = 1

        # for _ in range(v):

        #     lamda = 2

        #     for _ in range(lamda):

        #         # Iterate over all groups
        #         for i in topGroups_new:

        #             entitiesPresent = set()

        #             for ent in peerGroups[i[1]]:
        #                 entitiesPresent.add(ent[1])


        #             # Will replace the repth entity
        #             for rep in entitiesPresent:

        #                 for j in range(len(e_score)):
        #                     if j in entitiesPresent:
        #                         continue

        #                     newScore = 0

        #                     for alphaEnt in seed_inds:
        #                         newScore += 

        #                 #Check for better compatibility
        
        # PROBLEMS
        # 1. The Beta score is not dealt with correctly
        # entity_list = [[] for i in range(len(topGroups))]

        entity_list = {}
        for i in topGroups:
            temp = []
            for j in range(len(e_score)):
                if(entityTolist[j][i[1]] == 1):
                    temp.append(j)
            entity_list[i[1]] = temp

        for i in topGroups:
            # need to replace the last element, thats all
            temp = 0 #score obtained
            for j in peerGroups[i[1]]:
                temp += alphaScores[j[1]] + betaScores[j[1]]

            mexScore = -10000000
            repl = -1
            nInd = -1

            for j in peerGroups[i[1]]: # []
                # j = peerGroups[i[1]][j2]
                # j is the entity index
                # let me try to replace j in the peerGroup
                for k in entity_list[i[1]]:
                    if(tuple([alphaScores[k], k]) in peerGroups[i[1]]):
                        # print("Already in group")
                        continue 
                    # print("alpha scores : ", alphaScores[j[1]], alphaScores[k])
                    newScore = temp - alphaScores[j[1]] - betaScores[j[1]]
                    newScore += alphaScores[k]
                    newBetaScore = 0

                    for j1 in peerGroups[i[1]]:
                        if (j1[1] == j[1]):
                            continue
                        newBetaScore += beta * relMat[j1[1]][k]

                    if(len(peerGroups[i[1]])!=1):
                        newBetaScore /= (len(peerGroups[i[1]])-1)

                    newScore += newBetaScore

                    if mexScore < newScore and temp < newScore:
                        mexScore = newScore
                        repl = j
                        nInd = k


            if repl != -1 and nInd != -1:
                # print(peerGroups[i[1]], tuple([alphaScores[repl], repl]))
                # print(repl)
                peerGroups[i[1]].remove(repl)
                # print(nInd, alphaScores[nInd])
                peerGroups[i[1]].add(tuple([alphaScores[nInd], nInd]))
                # print(peerGroups[i[1]])

                print("Replacing", entities[repl[1]], "with", entities[k], "in facet", group_lists[i[1]])

        # for i in topGroups[:n_pg]:
        #     print(group_lists[i[1]], i[0])
        #     # print(peerGroups[i[1]])
        #     for j in peerGroups[i[1]]:
        #         print(entities[j[1]], j[0])
        #     print()
        # print(len(topGroups))
        
        # for i in done_set:
        #     print(group_lists[topGroups[i][1]], topGroups[i][0])
        #     for j in peerGroups[i]:
        #         print(entities[j[1]], j[0])
        #     print()
        for i in topGroups:
            if(i[1] in done_set):
                print(group_lists[i[1]], i[0])
                for j in peerGroups[i[1]]:
                    print(entities[j[1]], j[0])
                print()


        for i in topGroups:

            mex = -1000000000
            ind = -1
            for ent in entity_list[i[1]]:

                entScore = 0
                if tuple([alphaScores[ent], ent]) in peerGroups[i[1]]:
                    continue

                entScore += alphaScores[ent] 
                for p in peerGroups[i[1]]:
                    entScore += (beta/len(peerGroups[i[1]])) * relMat[p[1]][ent]

                kn = 0
                gammaScore = 0
                for d in done_set:
                    if d == i[1]:
                        continue

                    for ent_p in peerGroups[d]:
                        kn += 1
                        gammaScore += gamma * relMat[ent_p[1]][ent]

                entScore -= (gammaScore/kn)

                if mex < entScore:
                    mex = entScore
                    ind = ent

            print("adding element", entities[ind], "to facet", group_lists[i[1]])
            peerGroups[i[1]].add(tuple([alphaScores[ind], ind]))
        
        for i in topGroups:
            print(group_lists[i[1]], i[0])
            for j in peerGroups[i[1]]:
                print(entities[j[1]], j[0])
            print() 

 
 
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

        print("PROCESSING TIME:", time.time() - time_init)
 
        return peergroup_result
 
      

      # """
      # Problems to discuss with Neel:
      # 1. Update gamma score to the top groups? But that way the top k groups are penalised and the rest are not, that's fair?
      # 2. We will have to recalculate everything again, we need to do something smarter so as to compute only for the item to be replaced
      # """
