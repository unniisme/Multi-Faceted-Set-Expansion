#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:42:20 2019

@author: kpal
"""

from collections import defaultdict
import numpy as np
import math 

data_path = "/home/kpal/Documents/git-mpi/peering/src/"

def createdict(filepath):
    list_map = defaultdict(list)   # list --> list of entities
    entity_map = defaultdict(list) # entity --> list of sets
    list_dict = { }                # dictionary for list : id
    entity_dict = { }              # di
    e_count = 0
    l_count = 0
    
    with open(filepath, "r") as f:
        for line in f:
            elements = line.split("\t")
            entity = elements[0].strip()
            list_name = elements[1].strip("\n")
            if(entity in entity_map):
                entity_map[entity].append(list_name)
            else:
                entity_map[entity].append(list_name)
                entity_dict[entity] = e_count
                e_count += 1
            if(list_name in list_map):
                list_map[list_name].append(entity)
            else:
                list_map[list_name].append(entity)
                list_dict[list_name] = l_count
                l_count = l_count + 1
            
    return(list_dict, list_map, entity_dict, entity_map)

    
def filter_cat(filepath, cat_size):
    
    outfile = open(filepath, 'w')

    with open("/home/kpal/Documents/git-mpi/peering/src/data/categories.csv") as f:
        for line in f:
            lines = line.split("\",")
            group = lines[0].replace("\"http://yago-knowledge.org/resource/", "").strip()
            l_size = float(lines[1].strip())
            if(l_size > 4):
                outfile.write(group + "\t" + str(l_size) + "\n")
    outfile.close()
    
def filter_triples(filepath, e_candidates, l_candidates):
    
    outfile = open(filepath, 'w')

    with open("/home/kpal/Documents/git-mpi/peering/src/data/entities_categories.csv") as f:
        for line in f:
            lines = line.split("\t")
            entity = lines[0].strip()
            group = lines[1].strip()
            if(entity in e_candidates) and (group in l_candidates):
                outfile.write(entity + "\t" + group + "\n")
    outfile.close()

def read_size(filepath, elements, column):
    score = np.zeros(len(elements))
    with open(filepath, "r") as f:
        for line in f:           
            element = line.split("\t")
            e_name = element[0].strip()
            if(e_name in elements)==True:
                score[elements.index(e_name)] = float(element[column].strip())
    return(score)

   
def score_dict(filepath, elements, column):
    score = np.zeros(len(elements))
    with open(filepath, "r") as f:
        for line in f:
            element = line.split("\t")
            e_name = element[0].strip()
            if(e_name in elements)==True:
                score[elements.index(e_name)] = float(element[column].strip())
    for i in range(len(score)):
        if(score[i]>5):
            score[i] = math.log(score[i],2)
        else:
            score[i] = math.log(2,2)
    return(score)
    
def score_idf_dict(filepath, elements, column):
    score = np.zeros(len(elements))
    with open(filepath, "r") as f:
        for line in f:
            element = line.split("\t")
            e_name = element[0].strip()
            if(e_name in elements)==True:
                score[elements.index(e_name)] = float(element[column].strip())
    for i in range(len(score)):
        if(score[i]>0):
            score[i] = math.log(score[i],2)
            score[i] = 1/score[i]
    return(score)
            
    
def creatematrix(entity_map, entity_dict, list_dict):
    
    entityTolist_m = np.zeros((len(entity_dict), len(list_dict)))    
    for k in entity_map:
        for j in entity_map[k]:
                entityTolist_m[entity_dict[k],list_dict[j]] = 1.0
    
    return(entityTolist_m) 
    
def creatematrixWithCscore(entity_map, entity_dict, list_dict, catScore_dict):
    
    entityTolist_m = np.zeros((len(entity_dict), len(list_dict)))    
    for k in entity_map:
        for j in entity_map[k]:
            entityTolist_m[entity_dict[k],list_dict[j]] = catScore_dict[j]
    
    return(entityTolist_m)
    
def createdismatrix(Sim_m):
    
    m_shape = np.shape(Sim_m)
    dis_m= np.zeros(m_shape)
    for i in range(len(Sim_m)):
        for j in range(len(Sim_m[0])):
            dis_m[i,j]= 1-Sim_m[i,j]
    
    return(dis_m)
            
def createQuery(seedentities, entities):
    q_index = []
    q = np.zeros(len(entities))
    for i in seedentities:
        q[entities.index(i)] = 1
        q_index.append(entities.index(i))
    return(q, q_index)

def findAdditional(ex_seed, seeds):
    additionals= []
    for i in ex_seed:
        if (i in seeds)!=True :
            additionals.append(i)
    return(additionals)
    
def dictToArray(data_dict):
    data_array = [0]*len(data_dict)
    for i in data_dict:
        data_array[data_dict[i]] = i
    return(data_array)
    
def updateMatrix_3(ex_seed, seedentities, entityTolist_m, entities):
    
    common_list = []  
    additionals= findAdditional(ex_seed, seedentities)
            
    for i in range(len(entityTolist_m[0])):
        flag = True
        for k in ex_seed:
            if(entityTolist_m[entities.index(k),i]==0):
                flag = False
        if(flag == True):
            common_list.append(i)
    for i in common_list:
        for j in range(len(entityTolist_m)):
            entityTolist_m[j,i]=0
    
                
    return(common_list)  
    
def updateMatrix_2(ex_seed, seedentities, entityTolist_m, entities):
    
    common_list = []  
    additionals= findAdditional(ex_seed, seedentities)
            
    for i in range(len(entityTolist_m[0])):
        flag = True
        for k in ex_seed:
            if(entityTolist_m[entities.index(k),i]==0):
                flag = False
        if(flag == True):
            common_list.append(i)
    
            
    for k in additionals:        
        for i in range(len(entityTolist_m[0])):
            entityTolist_m[entities.index(k),i]=0
                
    return(common_list)  


def updateMatrix(ex_seed, seedentities, entityTolist_m, entities):

    common_list = []              
    for i in range(len(entityTolist_m[0])):
        flag = True
        for k in ex_seed:
            if(entityTolist_m[entities.index(k),i]==0):
                flag = False
        if(flag == True):
            common_list.append(i)
            for k in ex_seed:
                entityTolist_m[entities.index(k),i]=0
                        
    return(common_list)     

    
def findCommonList(ex_seed, entityTolist_m, entities):
    common_list = []      
            
    for i in range(len(entityTolist_m[0])):
        flag = True
        for k in ex_seed:
            if(entityTolist_m[entities.index(k),i]==0):
                flag = False
        if(flag == True):
            common_list.append(i)
                        
    return(common_list) 
    

                
def removeExSeeds(ex_seed, seedentities, entityTolist_m, entity_dict):
    common_list = [] 
    additionals= findAdditional(ex_seed, seedentities)            
    for i in range(len(entityTolist_m[0])):
        flag = True
        for k in ex_seed:
            if(entityTolist_m[entity_dict[k],i]==0):
                flag = False
        if(flag == True):
            common_list.append(i)                        
    for k in additionals:        
        for i in range(len(entityTolist_m[0])):
            entityTolist_m[entity_dict[k],i]=0           
    return(common_list)
                
     
def update(similarity_m, seed_index, peers_index, initial_q):
    """
    Changing similarity matrix dimensions 
    """
    print(seed_index, peers_index)
    additionals = []
    for i in peers_index:
       if (i in seed_index)!=True :
           additionals.append(i)

    similarity_m = np.delete(similarity_m, additionals, 0)
    similarity_m = np.delete(similarity_m, additionals, 1)
    initial_q = np.delete(initial_q, additionals, None)
    
    return(similarity_m, initial_q)

            
if __name__=='__main__':   
    
   list_dict, list_map, entity_dict, entity_map = createdict(data_path+"data/entities_categories.csv")
 
#   with open(data_path+'movie/entity_dict.json') as fp:
#       entity_dict = json.load(fp)
#             
#   with open(data_path+'movie/entity_map.json') as fp:
#       entity_map = json.load(fp)
#    
#   with open(data_path+'movie/list_dict.json') as fp:
#       list_dict = json.load(fp)
#
#    
#   entityTolist_m = creatematrix(entity_map, entity_dict, list_dict)
##    
#   sp.save_npz(data_path +"companies/etolist_matrix", sp.coo_matrix(entityTolist_m))    
#    
#   with open(data_path+'companies/entity_dict.json', 'w') as fp:
#        json.dump( entity_dict, fp)
#             
#   with open(data_path+'companies/entity_map.json', 'w') as fp:
#        json.dump( entity_map, fp)
#
#   with open(data_path+'companies/list_map.json', 'w') as fp:
#        json.dump( list_map, fp)
#        
#   with open(data_path+'companies/list_dict.json', 'w') as fp:
#        json.dump( list_dict, fp)
        
 #   print(len(list_map["wikicat_2004_television_series_endings"]))
#    initial_q = createQuery(["SH"], entity_dict)
#
#    sim_matrix = sc.createJSimMatrix(entityTolist_m)
#    print(entityTolist_m, end = "\n")
#    print(np.round(sim_matrix, decimals = 2), end = "\n")
#
#    updateMatrix(["SH", "AE", "AF"], ["SH"], entityTolist_m, entity_dict)
#    print(entityTolist_m, end = "\n")
#    sim_matrix = sc.createJSimMatrix(entityTolist_m)
#    print(np.round(sim_matrix, decimals = 2), end = "\n")
   

