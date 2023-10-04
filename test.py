from experiment_people import Model
import json

def load_files(path):
    with open(path) as json_file:
        return json.load(json_file)

folder_name = "data/people"

companies_entity_dict = load_files( folder_name + '/entity_dict.json')

companies_entity_map = load_files( folder_name + '/entity_map.json')

companies_list_dict = load_files( folder_name + '/list_dict.json')

companies_list_map = load_files( folder_name + '/list_map.json')

model = Model()
model.initiate(companies_entity_map, companies_list_map, companies_entity_dict, companies_list_dict)
model_result = model.calculate(folder_name, "Freddie_Mercury", 3, 5, 1, 3)