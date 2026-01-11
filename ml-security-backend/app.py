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
from interfaces.TrainTimeAttack import TrainTimeAttack
from interfaces.TrainTimeDefense import TrainTimeDefense

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

@app.route("/models",methods=["GET"])
def models():
    model_map=appContext.get_models()
    return jsonify(model_map)

@app.route("/run", methods=["POST"])
def run():
    payload=request.get_json()

    dataset = appContext.resolve_dataset(payload["dataset"])
    attack = appContext.resolve_attack(payload["attack"])
    model = appContext.resolve_model(payload["model"])
    defense = appContext.resolve_defense(payload["defense"])
    # metric = appContext.resolve_metric(payload["metric"])

    is_tt_defense = isinstance(defense, TrainTimeDefense)
    is_tt_attack = isinstance(attack, TrainTimeAttack)

    if is_tt_attack and is_tt_defense:
        raise Exception("Unable to run an online attack and online defense simultaneously.")

    globalHandler = GlobalHandler()
    globalHandler.register(DatasetHandler(dataset))
    globalHandler.register(AttackHandler(attack))
    if not (is_tt_defense or is_tt_attack):
        globalHandler.register(ModelHandler(model))
    globalHandler.register(DefenseHandler(defense))
    # globalHandler.register(MetricsHandler(metric))
    globalHandler.register(MetricsHandler())
    globalHandler.register(VisualizationHandler(num_samples=5))

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
            "attack": payload["attack"],
            "accuracy": context["acc"],
            "asr": context["acc_asr"]
        },
        "defense_phase": {
            "defense": payload["defense"],
            "accuracy": context["final_accuracy"],
            "asr": context["final_asr"]
        },
        "improvement": {
            "asr_reduction": context["acc_asr"] - context["final_asr"],
            "acc_drop": context["acc"] - context["final_accuracy"]
        },
        "visualizations": context.get("visualizations", [])
    }
    return jsonify(response)

if __name__=="__main__":
    app.run(debug=True)