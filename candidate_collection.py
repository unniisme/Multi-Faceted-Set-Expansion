#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:56:06 2019

@author: kpal
"""
data_path = "/home/kpal/Documents/git-mpi/peering/src/companies/"



import dataPrep as dp
import sim_var as sc
import numpy as np
import json
import scipy.sparse as sp
from collections import defaultdict


def candidate_collect(query, entity_map, list_map, depth, maxlimit):
    """
    This function takes a query, a mapping of entities to lists they belong to, a mapping of lists to entities, a depth, and a maximum limit.
    It collects a set of candidate entities based on the input query and the relationships between entities and lists.
    It starts with the query entities, then iteratively expands by including entities from lists that are related to the initially collected entities, up to a certain depth or limit.
    """
    final_candidates= []
    checked_lists = []
    total_entity = len(entity_map)
    limit = min(total_entity, maxlimit)
    temp_entities = query

    for i in range(depth):
        size = len(temp_entities)

        for j in range(size):
            if((len(final_candidates)+len(temp_entities))== limit):
                break
            poped_entity = temp_entities.pop(0)
            final_candidates.append(poped_entity)            
            temp_lists = entity_map[poped_entity]
            for k in temp_lists:
                if((len(final_candidates)+len(temp_entities))== limit):
                        break
                if ((k in checked_lists)!=True) and (len(list_map[k])<161):
                    ret_entity = list_map[k]
                    checked_lists.append(k)
                    for l in ret_entity:
                        if (((l in final_candidates) or (l in temp_entities))!=True):
                            temp_entities.append(l)
                            if((len(final_candidates)+len(temp_entities))== limit):
                                break
                    
           
    for e in temp_entities:
        final_candidates.append(e)     
    return(final_candidates, checked_lists)
    
def list_candidate_collect(candidates_entity, entity_map):
    # print(type(entity_map))
    # # candidates_entity = list(candidates_entity)
    # candidates_entity = candidates_entity[0]
    """
    This function takes a list of candidate entities and a mapping of entities to lists they belong to.
    It collects a set of candidate lists based on the input candidate entities by retrieving the lists associated with those entities.
    """
    list_candidates = []
    for i in candidates_entity:
        # print(i)
        lists = entity_map[i]
        for j in lists:
            if(j in list_candidates) != True:
                list_candidates.append(j)
    
    return(list_candidates)
            
def sliceMatrix(sim_matrix, candidates_index):
    """
    This function takes a similarity matrix and a set of candidate indices.
    It creates a new matrix by slicing the original similarity matrix based on the provided candidate indices, resulting in a submatrix.
    """
    
    size = len(candidates_index)
    slice_matrix = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            slice_matrix[i,j] = sim_matrix[candidates_index[i],candidates_index[j]]
            
    return(slice_matrix)

def sliceEtoList(entityTolist, candidate_l_index, candidate_index):
    """
    This function takes a matrix representing entity-to-list relationships, a set of candidate list indices, and a set of candidate entity indices.
    It creates a new matrix by slicing the original entity-to-list matrix based on the provided candidate indices for both lists and entities, resulting in a submatrix.
    """

    slice_matrix = np.zeros((len(candidate_index), len(candidate_l_index)))
    for i in range(len(candidate_index)):
        for j in range(len(candidate_l_index)):
            slice_matrix[i,j] = entityTolist[candidate_index[i], candidate_l_index[j]]
    return(slice_matrix)
    
def toyexample():
    list_dict, list_map, entity_dict, entity_map = dp.createdict("/home/kpal/Documents/git-mpi/peering/src/example.txt")
#    l_score = np.array([0.3,0.4, 0.3,0.8, 0.4, 0.3])
    entities = [0]*len(entity_dict)
    for i in entity_dict:
        entities[entity_dict[i]] = i
   
    group_lists = [0]*len(list_dict)
    for i in list_dict:
        group_lists[list_dict[i]] = i 
 
    entityTolist = dp.creatematrix(entity_map,entity_dict, list_dict)
    
    sim_matrix = sc.createJSimMatrix(entityTolist)
    print(np.round(sim_matrix, decimals = 2))
    query = ["NB"]
    
    candidates, explored_l = candidate_collect(query, entity_map, list_map, 2, 9)

    candidate_l = list_candidate_collect(candidates, entity_map)
      
    with open(data_path+'entities.json', 'w') as fp:
        json.dump( entities, fp)
    with open(data_path+'entity_map.json', 'w') as fp:
        json.dump( entity_map, fp)
    
    candidate_index = []
    for i in candidates:
        candidate_index.append(entities.index(i))
    
    candidate_index.sort()    
    candidate_l_index = []
    for i in candidate_l:
        candidate_l_index.append(group_lists.index(i))
    
    candidate_l_index.sort()    
    new_glist= []
    
    for i in candidate_l_index:
        new_glist.append(group_lists[i])
    
    new_S = sliceMatrix(sim_matrix, candidate_index)
    new_entityTolist = sliceEtoList(entityTolist, candidate_l_index, candidate_index)
    print(np.round(new_S, decimals = 2))
    print(new_S.shape, new_entityTolist.shape)
    print(new_entityTolist)
    print(new_glist)
    
if __name__=='__main__':
    
    with open(data_path+"queries/queries.json") as rf:
        query_entity = json.load(rf)
    
    candidate_q = defaultdict(list)
    
#    query = ["Friends"]
    with open(data_path+"entity_map.json") as rf:
        entity_map = json.load(rf)
        
    with open(data_path+"list_map.json") as rf:
        list_map = json.load(rf)
        
    with open(data_path+"entity_dict.json") as rf:
        entity_dict = json.load(rf)
        
    with open(data_path+"list_dict.json") as rf:
        list_dict = json.load(rf)
        
    #entityTolist = (sp.load_npz(data_path+"etolist_matrix_25k.npz")).toarray()
#    entityTolist = dp.creatematrix(entity_map, entity_dict, list_dict)
    
    for i in query_entity:
        query = []
        query.append(i)
        
        candidates, explored_l = candidate_collect(query, entity_map, list_map, 2, 2000)    
        candidate_q[i]= candidates
#        candidate_l = list_candidate_collect(candidates, entity_map)
    
    # with open(data_path+'queries/candidate_queries.json', "w") as rf:
    #     json.dump(candidate_q, rf)
        
#    candidate_index = []
#
#    for i in candidates:
#        candidate_index.append(entity_dict[i])    
#    candidate_index.sort()
#    
#    candidate_l_index = []
#    for i in candidate_l:
#        candidate_l_index.append(list_dict[i])    
#    candidate_l_index.sort()

#    entityTolist = sliceEtoList(entityTolist, candidate_l_index, candidate_index)
  
#    sp.save_npz(data_path +"movie/etolist_matrix_Lion_King", sp.csr_matrix(entityTolist))
    
    
    