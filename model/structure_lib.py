#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:59:15 2019

@author: kpal
"""

class PeerGroup:
    
    def initiate(self, group_id, peers, categories, s_score, co_score, d_score, group_label):
        self.group_id = group_id
        self.peers = peers
        self.categories = categories
        self.s_score = s_score
        self.co_score = co_score
        self.d_score = d_score
        self.group_label = group_label
    
    def pg_to_dict(self, pg):
        dict_ret = dict(group_id = pg.group_id, peers = pg.peers, categories = pg.categories, 
                        s_score = pg.s_score, co_score = pg.co_score, d_score = pg.d_score, group_label = pg.group_label)
        return dict_ret
    
    def dict_to_pg(self, dict):
        group_id = dict['group_id']
        peers = dict['peers']
        categories = dict['categories']
        s_score = dict['s_score']
        co_score = dict['co_score']
        d_score = dict['d_score']
        group_label = dict['group_label']
        pg = PeerGroup()
        pg.initiate(group_id, peers, categories, s_score, co_score, d_score, group_label)
        return pg

    
class Group:
    
    def initiate(self, group_id, peers, categories):
        self.group_id = group_id
        self.peers = peers
        self.categories = categories
    
    def pg_to_dict(self, g):
        dict_ret = dict(group_id = g.group_id, peers = g.peers, categories = g.categories)
        return dict_ret
    
    def dict_to_pg(self, dict):
        group_id = dict['group_id']
        peers = dict['peers']
        categories = dict['categories']
        g = Group()
        g.initiate(group_id, peers, categories)
        return g