#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 17:41:31 2019

@author: kpal
"""

import numpy as np
import model.dataPrep as dp
import scipy as sp

import collections

import logging

def weightedJaccard(a1, a2):
    nom = np.sum(np.minimum(a1, a2))
    denom =  np.sum(np.maximum(a1,a2))
    
    if(nom == 0 and denom == 0):
        return(0)
    else:
        return(nom/denom)

def weightedJaccardSparse(a1, a2):
    minimum = 0
    maximum = 0

    for key, value in a1.items():
        maximum += value
        if key in a2:
            minimum += min(a1[key], a2[key])

    maximum += sum(a2.values())

    nom = minimum
    denom = maximum - minimum

    if nom == 0 and denom == 0:
        return 0
    else:
        return nom / denom

def createWJSimMatrix(A, score):
    
    G = A*score
    number_entity = np.shape(G)[0]
    S = np.zeros((number_entity,number_entity ))
    
    for i in range(number_entity):
        for j in range(i, number_entity):
            S[i,j] = weightedJaccard(G[i],G[j])
            S[j,i] = S[i,j]
            
    return(S)

def pairwise_sparse_jaccard_distance_weighted(X):
    logging.info("")
    cx = sp.sparse.coo_matrix(X)
    sparse_dict = collections.defaultdict(dict)
    for i, j, v in zip(cx.row, cx.col, cx.data):

        sparse_dict[i][j] = v

    return sparse_dict

def createSimMatrix(A : np.ndarray, 
                    S : np.ndarray,
                    c_score : np.ndarray, 
                    e_score : np.ndarray, 
                    peer_group : list[int]
                    ) -> np.ndarray:
    """
    Create a similarity matrix based on input data.

    Parameters:
    A (numpy.ndarray): The input matrix representing some data.
    S (numpy.ndarray): The output similarity matrix to be computed.
    c_score (numpy.ndarray): Coefficient scores.
    e_score (numpy.ndarray): Entity scores.
    peer_group (list): List of indices representing a peer group.

    Returns:
    numpy.ndarray: The similarity matrix computed based on the input data.
    """
    logging.info("")
    logging.debug("A size : %s", A.shape)

    G = A*c_score
    number_entity = np.shape(G)[0]

    csr = sp.sparse.csr_matrix(G)
    custom_jaccard_distance_weighted = pairwise_sparse_jaccard_distance_weighted(csr)
    G_spars = custom_jaccard_distance_weighted

    if not peer_group:
        for i in range(number_entity):
            for j in range(i, number_entity):
                S[i, j] = (weightedJaccardSparse(G_spars[i], G_spars[j]) + (1 - abs(e_score[i] - e_score[j]))) / 2
                S[j,i] = S[i,j]

    else:
        for i in peer_group:
            for j in range(number_entity):
                S[i, j] = (weightedJaccardSparse(G_spars[i], G_spars[j]) + (1 - abs(e_score[i] - e_score[j]))) / 2
                S[j, i] = S[i, j]
            
    return(S)

def createJSimMatrix(G):
    
    S = (G.dot(G.T)).astype(float)
    sum_m = np.sum(G, axis = 1)

    for i in range(len(S)):
        for j in range(len(S[0:])):
            a =sum_m[i] + sum_m[j]
            if(a == 0):
                S[i,j] = 0
            else:
                S[i,j] = S[i,j]/(a-S[i,j])            
    return(S)
        
def pageRank(G, s, maxerr, q):
    
    rsum = np.sum(G, axis = 0)
    A = G/rsum[ None, : ]
    r = np.array([0.5,0,0.5,0,0,0,0,0,0,0,0])
    for count in range(100):
        prev_r = r.copy()
        r = (1-s)*A.dot(r.T) + s*q
        while (np.sum(abs(r-prev_r))< maxerr) == True:
            break        
    return r   
        
if __name__=='__main__':
        
    n_pg = 1  # number of peer groups
    n_p = 4   # number of peers in each group    
    Q = np.array([1,0,0,0,0,0,0,0,0,0,0])   
    entities = ["SH", "AE", "NT", "KS", "MK", "IN", "AF", "RT", "MP", "ES", "NB"]
    
    G = np.array([[1, 1, 1, 0, 1, 1],
       [1, 1, 0, 1, 1, 1],
       [1, 0, 1, 0, 0, 0],
       [1, 0, 1, 0, 0, 0],
       [0, 0, 1, 0, 0, 0],
       [0, 1, 0, 0, 1, 1],
       [1, 1, 0, 0, 1, 0],
       [0, 1, 0, 0, 1, 0],
       [0, 0, 0, 1, 0, 1],
       [0, 0, 0, 1, 0, 1],
       [0, 0, 0, 1, 0, 0]])
    
    e_score = np.array([0.9, 0.9, 0.7,0.7, 0.7, 0.8, 0.6, 0.6, 0.5, 0.6, 0.5, 0.5])
    l_score = np.array([0.3,0.4, 0.3,0.8, 0.4, 0.3])
    G_new = G*l_score  
        
#    S = np.around(createSimMatrix(G, e_score), decimals = 2)   
#    print(S[0]) 
#    S1 = np.around(createJSimMatrix(G), decimals = 2)   
#    print(S1[0])
    
    S_new = np.around(createWJSimMatrix(G_new), decimals = 2)
    print(S_new[0])
    rank = np.round(pageRank(S_new, 0.15, .001, Q), decimals = 3)    
    print(rank)    
    query = ["SH"]