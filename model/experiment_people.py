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
import pdb
import math

import logging
 
 
class Model:
 
 
    def initiate(self, entity_map, list_map, entity_dict, list_dict):
 
        self.entity_map = entity_map
        self.list_map = list_map
        self.entity_dict = entity_dict
        self.list_dict = list_dict
 
 
    def calculate(self, domain, query, n_pg = 5, n_p = 6, score_type = 1, category_score_type = 2, alpha = 0.3, beta = 0.3, gamma = 0.4, popFactor = 1):
 
        # print("NPG ",n_pg)
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
 
        candidates, cl = cancol.candidate_collect(copy.deepcopy(self.query),self.entity_map, self.list_map,2,2000)
 
        candidate_l = cancol.list_candidate_collect(candidates, self.entity_map)
 
        logging.debug("PRINTING SOMETHING")
        logging.debug("length of Candidates_l : %s", len(candidate_l))
        logging.debug("candidates_l : %s",candidate_l[0:5])
        logging.debug("candidates : %s",candidates[0:5])
 
        # enitityToList is a mapping  between entities and lists
        entityTolist = (sp.load_npz(self.data_path+"etolist_matrix.npz")).toarray()
        logging.debug("Entity to list size : %s", len(entityTolist))
        logging.debug("Entity to list[0] size : %s", len(entityTolist[0]))

    
        logging.debug("Entity to List mapping:\n%s", entityTolist[:10, :10])
        logging.info("loaded the matrix")
        candidate_index, candidate_l_index = [], []
 
        for i in candidates:
            candidate_index.append(self.entity_dict[i])
        candidate_index.sort()
 
        for i in candidate_l:
            candidate_l_index.append(self.list_dict[i])
        candidate_l_index.sort()


        entities = [0]*len(candidate_index)
        # print("CAND : ",candidates[:5])
 
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

        # isToy = 0
        #------------------------------


        # COMMENT BELOW LINES FOR TOY
        # ----------------------------
        entityTolist = cancol.sliceEtoList(entityTolist, candidate_l_index, candidate_index)
        isToy = 1
        # ----------------------------
 
        # a1, b1, a2, b2

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
        
        # BIG BUG
        # We arent updating topGroups of scores anywhere, full program
 
 #-------------------------------Beginning of Algorithm-----------------------------------------------

        # file_output = open('out.txt','a')

        # Hyper parameters       
        # alpha = 0.3
        # beta = 0.1
        # gamma = 0.6
 

        # Creating Relation Matrix
        time_init = time.time()
        relMat = np.around(sc.createSimMatrix(entityTolist, S, c_score, e_score, []), decimals = 2)
        time_init1 = time.time()

       
        entityTolistC = np.zeros((len(entityTolist), len(entityTolist[0])))
        # print("OUR ALGORITHM STARTS NOW")

        popScore = {}

        # Hyper Parameters
        # minGroups = 25
        # x = 3
        # thresholds = [[0.5,0.5,0.5,0.5],[0.95, 0.93, 0.9, 0.80]]
        thresholds = [[0.5,0.5,0.5,0.5],[0.90, 0.90, 0.90, 0.90]]
        minGroups = 25
        x = 3
        n_ele = x
        # popFactor = 1
        # -------------------


        """
        Good Result 
        thresholds = [[0.5,0.5,0.5,0.5],[0.95, 0.93, 0.9, 0.88]]
        minGroups = 15
        x = 3
    
        Even Better Result
        thresholds = [[0.5,0.5,0.5,0.5],[0.95, 0.93, 0.85, 0.80]]
        minGroups = 20
        x = 3
        Time = 15.276177406311035
        """

        for i in range(len(c_score)):
            if (math.log2(len(self.list_map[group_lists[i]])) ** 2) != 0:
                popScore[i] = popFactor/(math.log2(len(self.list_map[group_lists[i]])) ** 2)
            else:
                popScore[i] = 0
        # for i in range(10):
        #     print(group_lists[i], len(self.list_map[group_lists[i]]), popScore[i])

        # 
        # Initializing the group scores

        # Alpha Scores (Coherence with Seed)

        # 

        alphaScores = np.zeros(len(e_score))
        

        # print("SEED INDS : ", seed_inds)
        # print(entities[seed_inds[0]])
        for i in seed_inds:
            for k in range(len(c_score)):
                count = 0
                for j in range(len(e_score)):
                    if(j in seed_inds):
                        continue

                    if(entityTolist[j][k] == 1):
                        count += 1
                        entityTolistC[j][k] += ((alpha * relMat[i][j])/len(seed_inds))
                        alphaScores[j] = entityTolistC[j][k]
                if count!=0:
                    popScore[k] *= ((math.log2(count))**0.7)
                    # popScore[k] *= 1


        peerGroups = {}
        cnt = 0
 
        # 
 
        # STEP 1-2
        # STEP FOR FINDING HIGH x SCORES FOR EACH FACET
        for j in range(len(c_score)):        
            inds = set()
            inds_1 = []

            if(entityTolist[seed_inds[0]][j] == 0):
                continue

            # print(group_lists[j])
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

        # print("AFTER ALPHA ADDITION (ITERATION 0)")
 
        # print("Time:", time.time() - time_c)
        time_c = time.time()
 
        # Calculating Group Score
        groupScores = []
        # print("Top K groups")
        for i in range(len(c_score)):
            sm = 0
            if(i not in peerGroups):
                groupScores.append(0)
                continue
            for j in peerGroups[i]:
                sm += j[0] / x
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

        # 
        for i in range(len(c_score)):
            if(i in tgInd):
                continue
            # if(groupScores[i] >= (mex * 0.5 * (isToy) ) + (mex * thresholds[0] * (1 - isToy)) ):
            if(groupScores[i] >= (mex * thresholds[isToy][0]) ):

                topGroups.append([groupScores[i], i])
                tgInd.add(i)
        
        
        topGroups = sorted(topGroups, reverse=True)
        # print(topGroups[:n_pg])
        # for i in topGroups[:n_pg]:
        #     print(group_lists[i[1]], i[0])
        #     # print(peerGroups[i[1]])
        #     for j in peerGroups[i[1]]:
        #         print(entities[j[1]], j[0])
        #     print()

        # 
 
        # STEP 3-5
        # Beta Scores (Coherence among peer group)
        betaScores = np.zeros(len(e_score))
        # alphaScores = np.zeros(len(e_score))

        counts = np.zeros((len(entityTolist), len(entityTolist[0])))

        for k in topGroups:
            for j in peerGroups[k[1]]:
                for i in peerGroups[k[1]]:
                    if(i[1] == j[1]):
                        continue
                    
                    temp = x
                    groupScores[k[1]] += ((beta*(relMat[j[1]][i[1]]))) / (( (temp) * (temp - 1) )) 

                    if(temp != 1):
                        betaScores[i[1]] += beta * relMat[j[1]][i[1]] /  (temp - 1)
            # k[0] = groupScores[k[1]]

        # print("AFTER BETA ADDITION (ITERATION 0)")
        
 
        # print("Time:", time.time() - time_c)
        time_c = time.time()
 
        # Calculating Group Score
        # groupScores = []
        # print("Top K groups")
 
        topGroups_new = []
        tgInd = set()

        mex = -1000000000
        for i in topGroups:
            if(groupScores[i[1]] > mex):
                ind = i[1]
                mex = groupScores[i[1]]
            i[0] = groupScores[i[1]]

        topGroups_new.append([mex, ind])
        tgInd.add(ind)
        # 

        for i in topGroups:
            if(i[1] in tgInd):
                continue
            # if(groupScores[i[1]] >= (mex * 0.5 * (isToy) ) + (mex * thresholds[1] * (1 - isToy))):
            # 
            if(groupScores[i[1]] >= (mex * thresholds[isToy][1] ) ):

                topGroups_new.append([groupScores[i[1]], i[1]])
                tgInd.add(i[1])
 
        topGroups_new = sorted(topGroups_new, reverse=True)
        
        if (len(topGroups_new) < minGroups) and isToy:
            topGroups_new = sorted(topGroups, reverse=True)
            topGroups_new = topGroups_new[:minGroups]

        topGroups = copy.deepcopy(topGroups_new)
        # for i in topGroups[:n_pg]:
        #     print(group_lists[i[1]], i[0])
        #     # print(peerGroups[i[1]])
        #     for j in peerGroups[i[1]]:
        #         print(entities[j[1]], j[0])
        #     print()
        
        # 

        # Gamma Score (Dissimilarity)
        mexInd = -1
        mexScore = -10000

        for i in topGroups:
            if(i[0] + popScore[i[1]] > mexScore):
                mexScore = i[0] + popScore[i[1]]
                mexInd = i[1]

        # Ranking Groups
        topKGroups = []
        topKGroups.append([mexScore, mexInd])
        topKGroups_Ind = set()
        topKGroups_Ind.add(topKGroups[0][1])

        # 

        for i in range(n_pg - 1):
            top_score = -100000
            ind = -1            
            for j in topGroups:
                if j[1] in topKGroups_Ind:
                    continue
                temp = 0
                for ent in peerGroups[j[1]]:
                    for k in topKGroups:
                        for ents in peerGroups[k[1]]:
                            temp -= (gamma * (relMat[ent[1]][ents[1]]))

                # this may be wrong since the gamma calculation is weird
                # we should only take the top few groups? or all groups?
                # should this be 
                count = len(topKGroups) * x * x
                # 

                # count = (n_pg - 1) * (x)
                # print("count = ", count)
                if((j[0] + ((temp)/count) + popScore[j[1]]) > top_score):
                    top_score = j[0] + ((temp)/count) + popScore[j[1]]
                    ind = j[1]

            # print("Chose next facet", group_lists[ind])
            # print("Score Obtained : ", top_score)
            topKGroups.append([top_score, ind])
            topKGroups_Ind.add(ind)

        topKGroups = sorted(topKGroups, reverse = True)

        # print("AFTER GAMMA ADDITION (ITERATION 0)")
        
 
        # print("Time:", time.time() - time_c)
        time_c = time.time()

        # print("Top K groups (RANKED)")
       
        # pdb.set_trace()
        # for i in topKGroups:
        #     print(group_lists[i[1]], i[0])
        #     for j in peerGroups[i[1]]:
        #         print(entities[j[1]], j[0])
        #     print()


        # print("Number of groups left:", len(topGroups))
        for i in topKGroups:
            print(group_lists[i[1]])
            for j in peerGroups[i[1]]:
                print(entities[j[1]])
            print()

 
 ########################################################################################
 # Iterative Algorithm  
 ########################################################################################
 
        # Number of additions to group
        v = n_p - x

        entity_list = {}
        for i in topGroups:
            temp = []
            for j in range(len(e_score)):
                if(entityTolist[j][i[1]] == 1 and (j not in seed_inds)):
                    temp.append(j)
            entity_list[i[1]] = temp

      

        for _ in range(v):
            n_ele = x + _
            # Replace Only Once Until No change in reordered set 
            # print("Beginning Replacement in peer group (ITERATION", (_ + 1),")")

            lamda = 1
            count_loop = 0
            while count_loop < 10:
                # print()
                # print("Replacing elements in peer groups:", count_loop)
                count_loop += 1
                betaScores = {}

                for i in topGroups:
                    currScore = 0 #score obtained
                    for j in peerGroups[i[1]]:

                        betaScore = 0
                        for ji in peerGroups[i[1]]:
                            if ji[1] == j[1]:
                                continue
                            betaScore += beta * relMat[ji[1]][j[1]]


                        if n_ele!= 1:
                            betaScore /= (((n_ele)-1)*n_ele)

                        currScore += (alphaScores[j[1]]/n_ele) + betaScore
                        betaScores[j[1]] = betaScore

                    mexScore = 10000000
                    repl = -1
                    nInd = -1

                    for j in peerGroups[i[1]]:
                        if (alphaScores[j[1]]/n_ele) + betaScores[j[1]] < mexScore:
                            mexScore = (alphaScores[j[1]]/n_ele) + betaScores[j[1]]
                            repl = j[1]
                    
                    mexScore = currScore
                    # 
                    for k in entity_list[i[1]]:
                        if(tuple([alphaScores[k], k]) in peerGroups[i[1]]):
                            # Already in group
                            continue 

                        newScore = currScore - (alphaScores[repl]/n_ele) - (2*betaScores[repl])

                        newScore += (alphaScores[k]/n_ele)
                        newBetaScore = 0

                        for j1 in peerGroups[i[1]]:
                            if (j1[1] == repl):
                                continue
                            newBetaScore += beta * relMat[j1[1]][k]

                        if(n_ele!=1):
                            newBetaScore /= ((n_ele-1)*n_ele)

                        newScore += (2*newBetaScore)

                        if newScore > mexScore:
                            mexScore = newScore
                            nInd = k 
                            # 

                    # only alpha and beta
                    i[0] = currScore

                    if repl != -1 and nInd != -1:
                    
                        # 
                        
                        peerGroups[i[1]].remove(tuple([alphaScores[repl], repl]))
                        peerGroups[i[1]].add(tuple([alphaScores[nInd], nInd]))
                        i[0] = mexScore
                        # print("Replacing", entities[repl], "with", entities[nInd], "in facet", group_lists[i[1]])


                # Gamma Score (Dissimilarity)

                # topGroups = sorted(topGroups, reverse = True)
                # print()
                # print("REORDERING AFTER REPLACEMENT (ITERATION", (_+1),")")
                # Ranking Groups
                mexInd = -1
                mexScore = -10000

                for i in topGroups:
                    if(i[0] + popScore[i[1]] > mexScore):
                        mexScore = i[0] + popScore[i[1]]
                        mexInd = i[1]

                topKGroups = []
                topKGroups.append([mexScore, mexInd])
                topKGroups_Ind_new = set()
                topKGroups_Ind_new.add(topKGroups[0][1])

                # I think there is a mistake here
                # subtracting less for first groups, alot for the
                # other groups 
                for i in range(n_pg - 1):
                    top_score = -100000
                    ind = -1            
                    for j in topGroups:
                        if j[1] in topKGroups_Ind_new:
                            continue
                        temp = 0
                        count = 0
                        for ent in peerGroups[j[1]]:
                            for k in topKGroups:
                                for ents in peerGroups[k[1]]:
                                    temp -= (gamma * (relMat[ent[1]][ents[1]]))
                                    count += 1

                        count = (n_ele**2) * (len(topKGroups)) 
                        if(count == 0):
                            # print("Count = 0 for", group_lists[j[1]])
                            continue
                        if((j[0] + ((temp)/count) + popScore[j[1]]) > top_score):
                            top_score = (j[0] + ((temp)/count) + popScore[j[1]])
                            ind = j[1]

                    # print("Chose next facet", group_lists[ind])
                    # print("Score Obtained : ", top_score)
                    topKGroups.append([top_score, ind])
                    topKGroups_Ind_new.add(ind)

                topKGroups = sorted(topKGroups, reverse = True)

                # print("TOP K GROUPS (ITERATION",(_+1),")")
                # for i in topKGroups:
                #     print(group_lists[i[1]], i[0])
                #     # print(peerGroups[i[1]])
                #     for j in peerGroups[i[1]]:
                #         print(entities[j[1]], j[0])
                #     print()

                if topKGroups_Ind_new == topKGroups_Ind:
                    # print("NO CHANGES IN REORDERING \n Reordered", count_loop, "times")
                    break

                topKGroups_Ind = copy.deepcopy(topKGroups_Ind_new)
                topKGroups = sorted(topKGroups, reverse=True)


            # Final topKGroups indices are the ones after breaking from the while loop
            topKGroups_Ind = copy.deepcopy(topKGroups_Ind_new)

            print("Replacement Done")
            for i in topKGroups:
                print(group_lists[i[1]])
                for j in peerGroups[i[1]]:
                    print(entities[j[1]])
                print()
            
            # Replacement Done
            # Now Adding one new element and reordering 

            # print("ADDING NEW ELEMENT TO PEER GROUPS (ITERATION", (_+1), ")")
            groupsDone = set()
            topKGroups = sorted(topKGroups, reverse=True)
            topKGroups_new = []
            for k in topKGroups:
                j,i = k
                mex = -1000000000
                ind = -1
                minGammaScore = -10000000
                for ent in entity_list[i]:

                    entScore = 0
                    if tuple([alphaScores[ent], ent]) in peerGroups[i]:
                        continue

                    entScore += (alphaScores[ent]/(n_ele+1)) 
                    for p in peerGroups[i]:
                        entScore += ((2*beta)/(n_ele*(n_ele+1))) * relMat[p[1]][ent]

                    kn = 0
                    gammaScore = 0

                    for d in topKGroups:
                        if d[1] == i:
                            continue

                        for ent_p in peerGroups[d[1]]:
                            kn += 1
                            gammaScore += gamma * relMat[ent_p[1]][ent]

                    # kn = (n_pg - 1) * ((n_ele + 1)**2)
                    kn = (((len(topKGroups_new)) * (n_ele+1)) + ((n_pg - 1) * (n_ele)))*n_ele
                    entScore -= (gammaScore/kn)

                    minGammaScore = max(minGammaScore, gammaScore/kn)

                    if mex < entScore:
                        mex = entScore
                        ind = ent

                # 
                if (ind != -1):
                    # there are no elements to add
                    # make sure to subtract minGammaScore
                    # print("Got an element to add")
                    peerGroups[i].add(tuple([alphaScores[ind], ind]))

                else:
                    # mex = -minGammaScore
                    minTotScore = -1000000
                    count = 0
                    for k1 in peerGroups[i]:
                        totScore = alphaScores[k1[1]]/(n_ele+1)
                        for p in peerGroups[i]:
                            if p[1] == k1[1]:
                                continue 
                            totScore += ((2*beta)/(n_ele*(n_ele+1))) * relMat[p[1]][k1[1]]


                        gammaScore = 0
                        for k2 in topKGroups:
                            if(k2[1] == i):
                                continue
                            for k3 in peerGroups[k2[1]]:
                                gammaScore -= gamma * (relMat[k1[1]][k3[1]])
                                count += 1
                        # 
                        gammaScore /= (((len(topKGroups_new)) * (n_ele+1)) + ((n_pg - 1) * (n_ele)))*n_ele
                        totScore += gammaScore
                        minTotScore = max(minTotScore, totScore)

                    # print("No element to add, so adding ", minTotScore)
                    mex = minTotScore

                topKGroups_new.append([k[0] + mex, k[1]])
                groupsDone.add(k[1])

            topKGroups = copy.deepcopy(topKGroups_new)

            topKGroups = sorted(topKGroups, reverse=True)
            # 
            
            
            for i in topGroups:

                if i[1] in groupsDone:
                    continue

                groupsDone.add(i[1])

                mex = -1000000000
                ind = -1
                minGammaScore = -100000
                for ent in entity_list[i[1]]:

                    entScore = 0
                    if tuple([alphaScores[ent], ent]) in peerGroups[i[1]]:
                        continue

                    entScore += (alphaScores[ent]/(n_ele+1)) 
                    for p in peerGroups[i[1]]:
                        entScore += ((2*beta)/(n_ele*(n_ele+1))) * relMat[p[1]][ent]

                    kn = 0
                    gammaScore = 0


                    for d in topKGroups[:-1]:

                        for ent_p in peerGroups[d[1]]:
                            kn += 1
                            gammaScore += gamma * relMat[ent_p[1]][ent]

                    kn = (n_pg - 1) * ((n_ele + 1) ** 2)
                    entScore -= (gammaScore/kn)

                    minGammaScore = max(gammaScore/kn, minGammaScore)

                    if mex < entScore:
                        mex = entScore
                        ind = ent

                # currScore is score of the group
                currScore = i[0]
                for j in peerGroups[i[1]]:
                    kn = 0
                    gammaScore = 0
                    for d in topKGroups[:-1]:

                        for ent_p in peerGroups[d[1]]:
                            kn += 1
                            gammaScore += gamma * relMat[ent_p[1]][j[1]]

                    kn = (len(topKGroups)-1) * ((n_ele + 1)**2)
                    currScore -= (gammaScore/kn)
                
                if ind != -1:
                    # we found a new element
                    currScore += mex

                    # 

                    peerGroups[i[1]].add(tuple([alphaScores[ind], ind]))

                    # print("Adding", entities[ind], "to", group_lists[i[1]])

                else:
                    # we did not find a new element
                    # 
                    # pdb.set_trace()
                    minTotScore = -1000000
                    count = 0
                    for k1 in peerGroups[i[1]]:
                        totScore = alphaScores[k1[1]]/(n_ele+1)
                        for p in peerGroups[i[1]]:
                            if p[1] == k1[1]:
                                continue 
                            totScore += ((2*beta)/(n_ele*(n_ele+1))) * relMat[p[1]][k1[1]]

                        gammaScore = 0
                        for k2 in topKGroups:
                            if(k2[1] == i[1]):
                                continue
                            for k3 in peerGroups[k2[1]]:
                                gammaScore -= gamma * (relMat[k1[1]][k3[1]])
                                count += 1
                        # 
                        gammaScore /= ((n_pg - 1) * ((n_ele + 1)**2))
                        totScore += gammaScore
                        minTotScore = max(minTotScore, totScore)

                    # print("No element to add, so adding ", minTotScore)
                    currScore += minTotScore

                # print(topKGroups)

                # If we can replace the least scored top K group
                if (currScore + popScore[i[1]] > topKGroups[-1][0]) :
                    # 
                    # print(topKGroups)
                   
                    topKGroups_Ind.remove(topKGroups[-1][1])
                    topKGroups.pop()

                    # print("Replacing in top K groups:", group_lists[topKGroups[-1][1]], "with ", group_lists[i[1]])

                    topKGroups_Ind.add(i[1])
                    topKGroups.append([currScore + popScore[i[1]], i[1]])
                
                    # print(topKGroups)
                    topKGroups = sorted(topKGroups, reverse=True)


            
            # for i in topKGroups[:n_pg]:
            #     print(group_lists[i[1]], i[0])
            #     for j in peerGroups[i[1]]:
            #         print(entities[j[1]], j[0])
            #     print() 

            


            # RECALCULATING GROUP SCORES
            # print("RECALCULATING NEW GROUP SCORES")
            groupScores = {}
            # print("AFTER ALPHA ADDITION")
            
     
            # print("Time:", time.time() - time_c)
            time_c = time.time()
            # print("Top K groups")
            for i in topGroups:
                sm = 0
                for j in peerGroups[i[1]]:
                    # sm += entityTolistC[j[1]][i]
                    sm += alphaScores[j[1]] / (n_ele + 1)
                groupScores[i[1]] = sm 
                # bug here, should update topGroups also
                i[0] = sm
     
            topGroups_new = []
            tgInd = set()

            mex = -1000000000
            for i in topGroups:
                if(groupScores[i[1]] > mex):
                    ind = i[1]
                    mex = groupScores[i[1]]
            
            topGroups_new.append([mex, ind])
            tgInd.add(ind)

            # 
            
            for i in topGroups:
                if(i[1] in tgInd):
                    continue
                # if((groupScores[i[1]] >= (mex * 0.5 * (isToy) ) + (mex * 0.90 * (1 - isToy))) or (i[1] in topKGroups_Ind)):
                if(groupScores[i[1]] >= (mex * thresholds[isToy][2]) ):

                    topGroups_new.append([groupScores[i[1]], i[1]])
                    tgInd.add(i[1])
     
            topGroups_new = sorted(topGroups_new, reverse=True)
            if (len(topGroups_new) < minGroups) and isToy:
                topGroups_new = sorted(topGroups, reverse=True)
                topGroups_new = topGroups_new[:minGroups]

            topGroups = copy.deepcopy(topGroups_new)
            # print(topGroups[:n_pg])
            # for i in topGroups[:n_pg]:
            #     print(group_lists[i[1]], i[0])
            #     # print(peerGroups[i[1]])
            #     for j in peerGroups[i[1]]:
            #         print(entities[j[1]], j[0])
            #     print()

            # STEP 3-5
            betaScores = np.zeros(len(e_score))

            for k in topGroups:
                for j in peerGroups[k[1]]:
                    for i in peerGroups[k[1]]:
                        if(i[1] == j[1]):
                            continue
                        
                        temp = n_ele + 1
                        groupScores[k[1]] += ((beta*(relMat[j[1]][i[1]]))) / (( (temp) * (temp - 1) )) 

                        if(temp != 1):
                            betaScores[i[1]] += beta * relMat[j[1]][i[1]] /  (temp - 1)

                k[0] = groupScores[k[1]]

            # print("AFTER BETA ADDITION")
            
     
            # print("Time:", time.time() - time_c)
            time_c = time.time()
     
            # Calculating Group Score
            # groupScores = []
            # print("Top K groups")
            
     
            topGroups_new = []
            tgInd = set()

            mex = -1000000000
            for i in topGroups:
                i[0] = groupScores[i[1]]
                if(groupScores[i[1]] > mex):
                    ind = i[1]
                    mex = groupScores[i[1]]
            topGroups_new.append([mex, ind])
            tgInd.add(ind)

            for i in topGroups:
                if(i[1] in tgInd):
                    continue
                # if((groupScores[i[1]] >= (mex * 0.5 * (isToy) ) + (mex * 0.85 * (1 - isToy))) or (i[1] in topKGroups_Ind)):
                if(groupScores[i[1]] >= (mex * thresholds[isToy][3]) ):
                 
                    topGroups_new.append([groupScores[i[1]], i[1]])
                    tgInd.add(i[1])
     
            topGroups_new = sorted(topGroups_new, reverse=True)
            if (len(topGroups_new) < minGroups) and isToy:
                topGroups_new = sorted(topGroups, reverse=True)
                topGroups_new = topGroups_new[:minGroups]

            topGroups = copy.deepcopy(topGroups_new)
            # for i in topGroups[:n_pg]:
            #     print(group_lists[i[1]], i[0])
            #     for j in peerGroups[i[1]]:
            #         print(entities[j[1]], j[0])
            #     print()

            topKGroups = []
            # topKGroups.append(tuple(topGroups[0]))
            topKGroups.append([topGroups[0][0] + popScore[topGroups[0][1]], topGroups[0][1]])    
            topKGroups_Ind = set()
            topKGroups_Ind.add(topKGroups[0][1])

            # print("AFTER GAMMA ADDITION")
            
     
            # print("Time:", time.time() - time_c)
            time_c = time.time()

            for i in range(n_pg - 1):
                top_score = -100000
                ind = -1            
                for j in topGroups:
                    if j[1] in topKGroups_Ind:
                        continue
                    temp = 0
                    count = 0
                    for ent in peerGroups[j[1]]:
                        for k in topKGroups:
                            for ents in peerGroups[k[1]]:
                                temp -= gamma * (relMat[ent[1]][ents[1]])
                                count += 1

                    count = len(topKGroups) * ((n_ele + 1)**2)
                    if(count == 0):
                        # print("Count = 0 for", group_lists[j[1]])
                        continue
                    if((j[0] + ((temp)/count) + popScore[j[1]]) > top_score):
                        top_score = j[0] + ((temp)/count) + popScore[j[1]]
                        ind = j[1]

                # print("Chose next facet", group_lists[ind])
                # print("Score Obtained : ", top_score)
                topKGroups.append([top_score, ind])
                topKGroups_Ind.add(ind)

                for ii in topGroups:
                    if ii[1] == ind:
                        ii[0] = top_score

           
            # print("TOP K GROUPS RANKING")
            # for i in topKGroups:
            #     print(group_lists[i[1]], i[0])
            #     # print(peerGroups[i[1]])
            #     for j in peerGroups[i[1]]:
            #         print(entities[j[1]], j[0])
            #     print()

            # print("Number of groups left : ", len(topGroups))
            for i in topKGroups:
                print(group_lists[i[1]])
                for j in peerGroups[i[1]]:
                    print(entities[j[1]])
                print()

              

        # 
        
        count_loop = 0
        while True:
            # print()
            # print("Replacing elements in peer groups:", count_loop)
            count_loop += 1
            betaScores = {}

            for i in topGroups:
                currScore = 0 #score obtained
                for j in peerGroups[i[1]]:

                    betaScore = 0
                    for ji in peerGroups[i[1]]:
                        if ji[1] == j[1]:
                            continue
                        betaScore += beta * relMat[ji[1]][j[1]]

                    if(n_ele+1!=1):
                        betaScore /= ((n_ele)*(n_ele+1))

                    currScore += (alphaScores[j[1]]/(n_ele+1)) + betaScore
                    betaScores[j[1]] = betaScore

                mexScore = 10000000
                repl = -1
                nInd = -1

                for j in peerGroups[i[1]]:
                    if (alphaScores[j[1]]/(n_ele+1)) + betaScores[j[1]] < mexScore:
                        mexScore = (alphaScores[j[1]]/(n_ele+1)) + betaScores[j[1]]
                        repl = j[1]
                
                mexScore = currScore
                for k in entity_list[i[1]]:
                    if(tuple([alphaScores[k], k]) in peerGroups[i[1]]):
                        # Already in group
                        continue 

                    newScore = currScore - (alphaScores[repl]/(n_ele+1)) - (2*betaScores[repl])

                    newScore += (alphaScores[k]/(n_ele+1))
                    newBetaScore = 0

                    for j1 in peerGroups[i[1]]:
                        if (j1[1] == repl):
                            continue
                        newBetaScore += beta * relMat[j1[1]][k]

                    if(n_ele+1!=1):
                        newBetaScore /= ((n_ele+1-1)*(n_ele+1))

                    newScore += (2*newBetaScore)

                    if newScore > mexScore:
                        mexScore = newScore
                        nInd = k 
                        # 


                i[0] = currScore

                if repl != -1 and nInd != -1:
                
                    # 
                    
                    peerGroups[i[1]].remove(tuple([alphaScores[repl], repl]))
                    peerGroups[i[1]].add(tuple([alphaScores[nInd], nInd]))
                    i[0] = mexScore
                    # print("Replacing", entities[repl], "with", entities[nInd], "in facet", group_lists[i[1]])

            # 
            # Gamma Score (Dissimilarity)
            # topGroups = sorted(topGroups, reverse = True)
            # print()
            # print("REORDERING AFTER REPLACEMENT (ITERATION", (_+1),")")

            mexInd = -1
            mexScore = -10000

            for i in topGroups:
                if(i[0] + popScore[i[1]] > mexScore):
                    mexScore = i[0] + popScore[i[1]]
                    mexInd = i[1]
            # Ranking Groups
            topKGroups = []
            # topKGroups.append(tuple(topGroups[0]))
            topKGroups.append([mexScore, mexInd])

            topKGroups_Ind_new = set()
            topKGroups_Ind_new.add(topKGroups[0][1])


            for i in range(n_pg - 1):
                top_score = -100000
                ind = -1            
                for j in topGroups:
                    if j[1] in topKGroups_Ind_new:
                        continue
                    temp = 0
                    count = 0
                    for ent in peerGroups[j[1]]:
                        for k in topKGroups:
                            for ents in peerGroups[k[1]]:
                                temp -= (gamma * (relMat[ent[1]][ents[1]]))
                                count += 1
                    count = len(topKGroups) * ((n_ele + 1)**2)
                    if(count == 0):
                        # print("Count = 0 for", group_lists[j[1]])
                        continue
                    if((j[0] + ((temp)/count)) + popScore[j[1]] > top_score):
                        top_score = j[0] + ((temp)/count) + popScore[j[1]]
                        ind = j[1]

                # print("Chose next facet", group_lists[ind])
                # print("Score Obtained : ", top_score)
                topKGroups.append([top_score, ind])
                topKGroups_Ind_new.add(ind)

            topKGroups = sorted(topKGroups, reverse = True)

            print("TOP K GROUPS (ITERATION",(_+1),")")
            for i in topKGroups:
                print(group_lists[i[1]])
                for j in peerGroups[i[1]]:
                    print(entities[j[1]])
                print()

            if topKGroups_Ind_new == topKGroups_Ind:
                # print("NO CHANGES IN REORDERING \n Reordered", count_loop, "times")
                break

            topKGroups_Ind = copy.deepcopy(topKGroups_Ind_new)
            topKGroups = sorted(topKGroups, reverse=True)

        # file_output.write('ALPHA = ' + str(alpha) + ' BETA = ' + str(beta) + ' GAMMA = ' + str(gamma) + '\n')
        # for i in topKGroups:
        #     file_output.write(group_lists[i[1]] + " " + str(i[0]) + '\n')
        #     # print(peerGroups[i[1]])
        #     for j in peerGroups[i[1]]:
        #         file_output.write(entities[j[1]] + " " + str(j[0]) + '\n')
        #     file_output.write('\n')
        # file_output.write('--------------------------------------\n')

        # Final topKGroups indices are the ones after breaking from the while loop
        topKGroups_Ind = copy.deepcopy(topKGroups_Ind_new)  

        peerGroupResult = []
        groupID = 0
        for i in topKGroups:
            groupResult = {}
            groupResult['group_id'] = groupID
            groupID += 1

            peers = []
            for j in peerGroups[i[1]]:
                peers.append(entities[j[1]])

            groupResult['peers'] = peers 
            groupResult['categories'] = group_lists[i[1]]
            groupResult['group_label'] = group_lists[i[1]]

            peerGroupResult.append(groupResult)

 
        # print("done calculating peer group")
        # peergroup_result[self.query[0]] = [peergroupList]
        peergroup_result[self.query[0]] = []
 
        time_d = time.time() - time_c
        processing_time.append(time_d)
 
        count = 0
        # for i in setup_time:
        #     count += i
        # print("avg setup time: ", count / len(self.query))
 
        # count = 0
        # for i in processing_time:
        #     count += i
        # print("avg setup time: ", count / len(self.query))
        print("PROCESSING TIME FOR RELMAT:", time_init1 - time_init)
        print("PROCESSING TIME:", time.time() - time_init)
        
        return peerGroupResult
 
      
# Replace until no change in reordering

"""

Report
1. Replace Algo o_o
2. Examples o_o
3. Appendix o_o
4. Ground truth !
9. Parameter Tuning o_o
5. mTurk, b3 o_o
6. Tabular information o_o
7. FUSE results comparision !
8. Presentation


def calculateAlphaScore():


reorderGroups()


2 tables
1 example
score table -> apna, fasets, seisa, rw, fuse

Time -> also


3,10 5,10, 3,5
Analysis (runtime) write
example
groundtruth

"""