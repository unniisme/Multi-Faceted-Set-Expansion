from flask import Flask, render_template, request
from model.experiment_people import Model
import json

app = Flask(__name__)
modelpath = "data/people"
datafile = modelpath + "/entity_dict.json"

#functions
def load_files(path):
    with open(path) as json_file:
        return json.load(json_file)

def initiate_model(folder_name):

    companies_entity_dict = load_files( folder_name + '/entity_dict.json')

    companies_entity_map = load_files( folder_name + '/entity_map.json')

    companies_list_dict = load_files( folder_name + '/list_dict.json')

    companies_list_map = load_files( folder_name + '/list_map.json')

    model = Model()
    model.initiate(companies_entity_map, companies_list_map, companies_entity_dict, companies_list_dict)
    print("Initiated model.")
    return model


# Initialization
data = load_files(datafile)
model = initiate_model(modelpath)
entries = data.keys()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        selected_entry = request.form.get("entry")
        num_categories = int(request.form.get("num_categories"))
        num_elements = int(request.form.get("num_elements"))

        res = model.calculate(modelpath, selected_entry, num_categories, num_elements)
        
        return render_template("index.html", entries=entries, result=res, selected_entry=selected_entry, num_categories=num_categories, num_elements=num_elements)

    return render_template("index.html", entries=entries, result=None, selected_entry=None, num_categories=None, num_elements=None)

if __name__ == "__main__":
    app.run(debug=True)