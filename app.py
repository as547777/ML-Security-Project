from flask import Flask, request, jsonify

from context.plugin_loader import load_plugins
from context.AppContext import AppContext

from handler.AttackHandler import AttackHandler
from handler.DatasetHandler import DatasetHandler
from handler.DefenseHandler import DefenseHandler
from handler.GlobalHandler import GlobalHandler
from handler.MetricsHandler import MetricsHandler
from handler.ModelHandler import ModelHandler

import attack as attack_pkg
import deffence as deffence_pkg
import dataset as dataset_pkg
import model as model_pkg
import statistic as metric_pkg


load_plugins(attack_pkg)
load_plugins(deffence_pkg)
load_plugins(dataset_pkg)
load_plugins(model_pkg)
load_plugins(metric_pkg)

appContext= AppContext()


app=Flask(__name__)

@app.route("/test", methods=["GET"])
def testPolymorphism():
    response = appContext.resolve_attack("ConcreteAttack")
    return response.execute("123", "123")


@app.route("/run", methods=["POST"])
def run():
    payload=request.get_json()
    
    dataset = appContext.resolve_dataset(payload["dataset"])
    attack=appContext.resolve_attack(payload["attack"])
    defense= appContext.resolve_defense(payload["defense"])
    metric= appContext.resolve_metric(payload["metric"])
    model= appContext.resolve_defense(payload["model"])

    globalHandler = GlobalHandler()
    globalHandler.register(DatasetHandler(dataset))
    globalHandler.register(ModelHandler(model))
    globalHandler.register(DefenseHandler(defense))
    globalHandler.register(AttackHandler(attack))
    globalHandler.register(MetricsHandler(metric))

    context={"learning_rate" : payload["learning_rate"],
             "epochs": payload["epochs"]}
    
    results=globalHandler.handle(context)

    return jsonify(results)


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