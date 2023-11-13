from flask import Flask, render_template, request, flash
from model.experiment_people import Model
import json
import os

app = Flask(__name__)
basepath = "data/"

#functions
def load_files(path):
    with open(path) as json_file:
        return json.load(json_file)
    

def initiate_model(folder_name):

    entity_dict = load_files( folder_name + '/entity_dict.json')

    entity_map = load_files( folder_name + '/entity_map.json')

    list_dict = load_files( folder_name + '/list_dict.json')

    list_map = load_files( folder_name + '/list_map.json')

    model = Model()
    model.initiate(entity_map, list_map, entity_dict, list_dict)
    print(f"Initiated model at {folder_name}")
    return model

# Initialization
models = {}
for directory in os.listdir(basepath):
    datafile = basepath + directory + "/entity_dict.json"
    modelpath = basepath + directory

    data = load_files(datafile)
    model = initiate_model(modelpath)
    model.load_similarity_matrix(modelpath+"/")
    entries = data.keys()

    models[directory] = {"model" : model, "entries" : entries, "modelpath" : modelpath}

print("Finished intializing")

@app.route("/", methods=["GET", "POST"])
def select_page():
    return render_template("select.html", models = models)

@app.route("/models/<model_name>", methods=["GET", "POST"])
def index(model_name):
    model = models[model_name]["model"]
    entries = models[model_name]["entries"]
    modelpath = models[model_name]["modelpath"]

    precomputed_present = os.path.isfile(modelpath + "/sim_matrix.npz")

    if request.method == "POST":
        selected_entry = request.form.get("entry")
        num_categories = int(request.form.get("num_categories"))
        num_elements = int(request.form.get("num_elements"))

        precomputed = bool(request.form.get("precomputed", False))
            

        res = model.calculate(modelpath, selected_entry, num_categories, num_elements, use_precalculated=precomputed)
        
        return render_template("index.html", 
                               entries=entries, result=res, selected_entry=selected_entry, 
                               num_categories=num_categories, num_elements=num_elements,
                               precomputed_present=precomputed_present)

    return render_template("index.html", entries=entries, result=None, selected_entry=None, 
                           num_categories=None, num_elements=None, precomputed_present=precomputed_present)

if __name__ == "__main__":
    app.run(debug=True)
