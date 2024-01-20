from flask import Flask, render_template, request, redirect, url_for
from model.experiment_people import Model
import json
import os

app = Flask(__name__)
# Initialization
basepath = "data/"
models = {}

#functions
def load_file(path):
    with open(path) as json_file:
        return json.load(json_file)
    
def save_file(path, obj):
    with open(path, "w") as json_file:
        json.dump(obj, json_file)
    

def initiate_model(folder_name):

    model = Model()

    # entity_map has to necessarily be present

    try:
        entity_map = load_file( folder_name + '/entity_map.json')
    
        entity_dict = load_file( folder_name + '/entity_dict.json')

        list_dict = load_file( folder_name + '/list_dict.json')

        list_map = load_file( folder_name + '/list_map.json')

    except OSError as e:
        # Only entity_map present
        entity_map, list_map, entity_dict, list_dict = model.create_map(folder_name + '/entity_map.json')

        save_file( folder_name + '/entity_dict.json', entity_dict)

        save_file( folder_name + '/list_dict.json', list_dict)

        save_file( folder_name + '/list_map.json', list_map)
    
    model.initiate(entity_map, list_map, entity_dict, list_dict)
    
    print(f"Initiated model at {folder_name}")
    return model

def load_models():
    for directory in os.listdir(basepath):
        if directory in models:
            continue

        modelpath = basepath + directory

        model = initiate_model(modelpath)
        model.load_similarity_matrix(modelpath+"/")
        
        datafile = basepath + directory + "/entity_dict.json"
        data = load_file(datafile)
        entries = data.keys()

        models[directory] = {"model" : model, "entries" : entries, "modelpath" : modelpath}


load_models()
print("Finished intializing")

@app.route("/", methods=["GET", "POST"])
def select_page():
    
    if request.method == "POST":
        # Get the uploaded file and group name from the form
        uploaded_file = request.files['file']
        dir_name = request.form['groupName']

        # Check if a file was selected
        if uploaded_file.filename == '':
            return "No file selected!"

        # Save the file to a folder
        os.makedirs(basepath + dir_name, exist_ok=True)
        file_path = os.path.join(basepath + dir_name, "entity_map.json")
        uploaded_file.save(file_path)

        try:
            load_models()
        except Exception as e:
            return str(e)

        return redirect(request.url)

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
