from flask import Flask, request, jsonify
from flask_cors import CORS

from context.plugin_loader import load_plugins
from context.AppContext import AppContext

from handler.AttackHandler import AttackHandler
from handler.DatasetHandler import DatasetHandler
from handler.DefenseHandler import DefenseHandler
from handler.GlobalHandler import GlobalHandler
from handler.MetricsHandler import MetricsHandler
from handler.ModelHandler import ModelHandler
from handler.VisualizationHandler import VisualizationHandler

import attack as attack_pkg
import defence as defence_pkg
import dataset as dataset_pkg
import model as model_pkg
import statistic as metric_pkg
from utils.dict_to_list import dict_to_list

load_plugins(attack_pkg)
load_plugins(defence_pkg)
load_plugins(dataset_pkg)
load_plugins(model_pkg)
load_plugins(metric_pkg)

appContext= AppContext()
app=Flask(__name__)
CORS(app)

@app.route("/attacks", methods=["GET"])
def attacks():
    return jsonify(dict_to_list(appContext.fetch_attacks()))

@app.route("/defenses", methods=["GET"])
def defenses():
    return jsonify(dict_to_list(appContext.fetch_defenses()))

@app.route("/datasets", methods=["GET"])
def datasets():
    return jsonify(dict_to_list(appContext.fetch_datasets()))

@app.route("/run", methods=["POST"])
def run():
    payload=request.get_json()

    dataset = appContext.resolve_dataset(payload["dataset"])
    attack = appContext.resolve_attack(payload["attack"])
    model = appContext.resolve_model(payload["model"])
    defense = appContext.resolve_defense(payload["defense"])
    # metric = appContext.resolve_metric(payload["metric"])

    globalHandler = GlobalHandler()
    globalHandler.register(DatasetHandler(dataset))
    globalHandler.register(AttackHandler(attack))
    globalHandler.register(ModelHandler(model))
    globalHandler.register(DefenseHandler(defense))
    # globalHandler.register(MetricsHandler(metric))
    globalHandler.register(VisualizationHandler(num_samples=5))

    # context = {}
    context={"learning_rate" : payload["learning_rate"],
             "dataset": payload["dataset"],
             "epochs": payload["epochs"],
             "momentum": payload["momentum"],
             "attack_params": payload.get("attack_params",{}),
             "defense_params": payload["defense_params"],
             "model": model,
             "attack_instance": attack
    }

    # TODO - ovdje umjesto cijelog contexta vratiti samo metrics dio
    results=globalHandler.handle(context)
    # return jsonify(results)
    
    response = {
        "attack_phase": {
            "accuracy": context["acc"],
            "asr": context["acc_asr"]
        },
        "defense_phase": {
            "acc_pruned": context.get("acc_clean_pruned",""),
            "asr_pruned": context.get("acc_asr_pruned",""),
            "accuracy": context["final_accuracy"],
            "asr": context["final_asr"]
        },
        "improvement": {
            # Koliko smo smanjili uspješnost napada (što veće to bolje)
            "asr_reduction": context["acc_asr"] - context["final_asr"],
            # Koliko smo izgubili na točnosti (što bliže 0 to bolje)
            "acc_drop": context["acc"] - context["final_accuracy"]
        },
        "visualizations": context.get("visualizations", [])
    }

    if "ban_results" in results:
        ban_results = results["ban_results"]
        response["ban_detection"] = {
            "backdoor_detected": ban_results["backdoor_detected"],
            "original_accuracy": ban_results.get("original_accuracy", 0),
            "perturbed_accuracy": ban_results.get("perturbed_accuracy", 0),
            "accuracy_drop": ban_results.get("original_accuracy", 0) - ban_results.get("perturbed_accuracy", 0),
            "detection_time": ban_results.get("detection_time", 0),
            "positive_loss": ban_results.get("positive_loss", 0),
            "negative_loss": ban_results.get("negative_loss", 0),
            "fine_tuned": ban_results.get("fine_tuned", False),
            "fine_tuned_accuracy": ban_results.get("fine_tuned_accuracy", None),
            "error": ban_results.get("error", None)
        }
    return jsonify(response)

@app.route("/dummy/response", methods=["POST"])
def dummy_response():
    payload=request.get_json()
    
    result={"response_message": "Hello from backend!",
        "learning_rate": payload["learning_rate"],
            "epochs": payload["epochs"],
            "dataset": payload["dataset"],
            "attack": payload["attack"],
            "defense": payload["defense"]
            }
    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True)