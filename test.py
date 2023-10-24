from model.experiment_people import Model
import json
import logging
import sys

def load_files(path):
    with open(path) as json_file:
        return json.load(json_file)

folder_name = "data/companies"

companies_entity_dict = load_files( folder_name + '/entity_dict.json')

companies_entity_map = load_files( folder_name + '/entity_map.json')

companies_list_dict = load_files( folder_name + '/list_dict.json')

companies_list_map = load_files( folder_name + '/list_map.json')

model = Model()
model.initiate(companies_entity_map, companies_list_map, companies_entity_dict, companies_list_dict)
# print("Precomputing full similarity matrix")
# model.precompute_similarity(folder_name)
print("Initiated model.")
print("Calculating for Four_Points_by_Sheraton")
model_result = model.calculate(folder_name, "Four_Points_by_Sheraton", 3, 5, use_precalculated=len(sys.argv)>1)