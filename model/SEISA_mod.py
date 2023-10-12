#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:49:37 2019

@author: kpal
"""


import numpy as np
import model.dataPrep as dp
import model.sim_var as sc
import model.structure_lib as util
import json


def checklist(l1, l2):
    flag = True
    for i in l1:
        if (i in l2)!=True :
            flag = False
            break
    return(flag)

def findGroupinL(initial_q, S, n_p, B, list_map, group_lists):
    """
    Here we consider the condition that the peer group must be subset of original input data
     and thus secure the stucture of peer group
    """
    relSim = (initial_q.dot(S))/initial_q.sum()
#    print(relSim)
    prevRank = (np.argsort(relSim)[-n_p:]).tolist()
 #   print(prevRank)    
    while(True):
        coh_q = np.zeros([len(S)])
        for i in prevRank:
            coh_q[i]=1
        
        cohSim = (coh_q.dot(S))/n_p
        final_score = (relSim+cohSim)/2
        tempRank = (np.argsort(final_score)[-n_p:]).tolist()
        numberOflist = len(B[0])

        list_score = np.zeros([numberOflist])
        for i in range(numberOflist):
            for j in tempRank:
                list_score[i] = list_score[i]+ (B[j,i]*final_score[j])
                
        maxindex = np.argmax(list_score)       
        maxlist = np.multiply(final_score, B[:,maxindex] )        
        newRank = (np.argsort(maxlist)[-n_p:]).tolist()
#        print("newRank:", newRank)      
        if (checklist(prevRank, newRank)):
 #       if ((checklist(prevRank, newRank)) and (len(list_map[cat])>=n_p)):           
            print(maxindex)
            break        
        else:     
#            print("did not matched", len(list_map[cat]))
            prevRank.pop(0)
            temp = newRank[::-1]
            for i in temp:
                if (i in prevRank) != True :
                    prevRank.append(i)
                    break
            
    return(prevRank, maxindex)
         

def findGroup_Listconst_discomp(initial_q, S, n_p, B, pp, D, list_map, group_lists):
    """
    Here we consider the condition that the peer group must be a subset of original input data
     and thus secure the stucture of peer group      
     Additionally consider the dissimilarity function where dissimilarity score is nothing 
     but the negative similarity score of an entity with previously found peers     
    """
    relSim = (initial_q.dot(S))/initial_q.sum()
    prevRank = (np.argsort(relSim)[-n_p:]).tolist()
    seen_index = []
    dis_q = np.zeros([len(S)])
    
    if(len(pp)==0):
        dis = np.zeros([len(S)])
    else:
        for i in pp:
            dis_q[i]=1        
        dis = (dis_q.dot(D))/len(pp)

    while(True):
#        print("prevRank:", prevRank )
        coh_q = np.zeros([len(S)])
        for i in prevRank:
            coh_q[i]=1
            
        cohSim = (coh_q.dot(S))/n_p        
        final_score = (0.25*relSim)+(0.25*cohSim)+(0.5*dis)
        tempRank = (np.argsort(final_score)[-n_p:]).tolist()      
        numberOflist = len(B[0])
        list_score = np.zeros([numberOflist])
        for i in range(numberOflist):
            
            for j in tempRank:
                list_score[i] = list_score[i]+ (B[j,i]*final_score[j])
            
        rank_list_index = np.argsort(np.array(list_score))        
        maxindex = 0
        
        for i in reversed(range(len(list_score))):
            checking = rank_list_index[i]
            if(checking in seen_index)!=True:
                seen_index.append(checking)
                maxindex = checking
                break        
 #       maxindex = np.argmax(list_score)
        maxlist = np.multiply(final_score, B[:,maxindex] )        
        newRank = (np.argsort(maxlist)[-n_p:]).tolist()
#        print("newRank:", newRank)       
         
        cat = group_lists[maxindex]
        if ((checklist(prevRank, newRank)) and (len(list_map[cat])>=n_p)):  
            sim_score = 0
            co_score = 0
            distance = 0
            for i in newRank:
                sim_score = sim_score+ relSim[i]
                co_score = co_score + cohSim[i]
                distance = distance + dis[i]
            print(sim_score/n_p, co_score/n_p, distance/n_p)

            break       
        else:     
#            print("did not matched", len(list_map[cat]))
            prevRank.pop(0)
            temp = newRank[::-1]
            for i in temp:
                if (i in prevRank) != True :
                    prevRank.append(i)                    
                    break
            
    return(prevRank, maxindex)
    
    
def findGroup_Listconst_discomp_score(initial_q, S, n_p, B, pp, D, list_map, group_lists, list_score, found_list):
    """
    Here we consider the condition that the peer group must be a subset of original input data
     and thus secure the stucture of peer group, Additionally consider the dissimilarity function where
     dissimilarity score is nothing but the negative similarity score of an entity with previously found peers,
    but list prioritization has been done using list scores.   
    """
    iteration_check_list = [n_p]*len(group_lists)
    relSim = (initial_q.dot(S))/initial_q.sum()
    prevRank = (np.argsort(relSim)[-n_p:]).tolist()
    seen_index = []
    dis_q = np.zeros([len(S)])
    
    if(len(pp)==0):
        dis = np.zeros([len(S)])
    else:
        for i in pp:
            dis_q[i]=1        
        dis = (dis_q.dot(D))/len(pp)

    while(True):
        # print("prevRank:" , prevRank)
        coh_q = np.zeros([len(S)])
        for i in prevRank:
            coh_q[i]=1
         
        """
V1 0.25 0.25 .5, V2 0.3 0.3 .4 V3 0.2 0.2 0.6        """
        cohSim = (coh_q.dot(S))/n_p        
        final_score = (0.25*relSim)+(0.25*cohSim)+(0.5*dis)
        tempRank = (np.argsort(final_score)[-n_p:]).tolist()      
        numberOflist = len(B[0])
        
        temp_list_score = np.zeros(numberOflist)
        for i in range(numberOflist):
            
            for j in tempRank:
                temp_list_score[i] = temp_list_score[i]+ (B[j,i])
            temp_list_score[i] = temp_list_score[i]+ list_score[i]    
            
        for i in found_list:
            temp_list_score[i] = 0
            
        rank_list_index = np.argsort(temp_list_score)     
        
        for i in reversed(range(len(list_score))):
            checking = rank_list_index[i]
            iteration_check_list[checking] -= 1
            if(checking not in seen_index):
                if (len(list_map[group_lists[checking]])>=n_p) and (iteration_check_list[checking]>0):
                    maxindex = checking
                    break
                else:
                    seen_index.append(checking)

        maxlist = np.multiply(final_score, B[:,maxindex] )        
        newRank = (np.argsort(maxlist)[-n_p:]).tolist()
        # print("newRank:", newRank)       
         
        cat = group_lists[maxindex]
        if ((checklist(prevRank, newRank)) and (len(list_map[cat])>=n_p)):  
            sim_score = 0
            co_score = 0
            distance = 0
            for i in newRank:
                sim_score = sim_score+ relSim[i]
                co_score = co_score + cohSim[i]
                distance = distance + dis[i]
            
            group_score = (sim_score/n_p, co_score/n_p, distance/n_p)           
            break       
        else:     
            prevRank.pop(0)
            temp = newRank[::-1]
            for i in temp:
                if (i in prevRank) != True :
                    prevRank.append(i)                    
                    break    
   
    return(prevRank, maxindex, group_score)
    

if __name__=='__main__':
    data_path = "/home/kpal/Documents/git-mpi/peering/src/"
    query = ["According_to_Jim"]
    n_pg, n_p = 5, 11   # number of peer groups , number of peers in each group
    
    list_dict, list_map, entity_dict, entity_map = dp.createdict(data_path+"sitcoms/filtered_etocat.csv")
    
    entities = dp.dictToArray(entity_dict)  
    group_lists = dp.dictToArray(list_dict)  
        
    c_score = dp.score_dict(data_path+"sitcoms/categorieslangs.csv", group_lists,1)
    c_score = np.true_divide(c_score, np.amax(c_score))
    
    catsize_score = dp.score_dict(data_path+"sitcoms/categories_size.csv", group_lists,1)
    catsize_score_norm = np.true_divide(1,catsize_score)
    
    e_score = dp.score_dict(data_path+"sitcoms/entities_views.csv",entities,1)
    e_score = np.true_divide(e_score, np.amax(e_score))
    
    entityTolist = dp.creatematrix(entity_map, entity_dict, list_dict)

    initial_q, q_index = dp.createQuery(query, entity_dict)
    found_peers = []
     
    print(found_peers)
     
         
#    with open("/home/kpal/Documents/git-mpi/peering/scientists.json", 'w') as pfile:
#        json.dump(entities, pfile, ensure_ascii=False)
#    with open("/home/kpal/Documents/git-mpi/peering/categories.json", 'w') as lfile:
#        json.dump(group_lists, lfile, ensure_ascii=False)    
        
    for i in range(n_pg):      

#        S = np.around(sc.createJSimMatrix(entityTolist_m), decimals = 2)
        S = np.around(sc.createSimMatrix(entityTolist, c_score, e_score), decimals = 2)
        D = dp.createdismatrix(S)
#        print(S)
        
        peer_group, c_index = findGroup_Listconst_discomp(initial_q, S, n_p, entityTolist, found_peers, D, list_map, group_lists)
#        peer_group, c_index = findGroupinL(initial_q, S, n_p, entityTolist_m, list_map, group_lists)
        print("peer group" , i , " : ", group_lists[c_index])
        group = []
                
        for j in peer_group:
            if ((j in found_peers)!= True):
                found_peers.append(j)                
            group.append(entities[j])
        print(group)
        print("found peers are:", found_peers)
   
        common_list = dp.updateMatrix(group, query, entityTolist, entities)
  #      common_list = dp.findCommonList(group, entityTolist_m, entity_dict)
        for j in common_list:
            print(group_lists[j])    
