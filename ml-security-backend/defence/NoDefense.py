from interfaces.AbstractDefense import AbstractDefense

class NoDefense(AbstractDefense):
    __desc__ = {
        "display_name": "None",
        "description": "No defense mechanism will be carried out.",
        "type": None,
        "time": None,
        "params": {}
    }

    def execute(self, model, data, params, context):
        return {"final_accuracy": 0,
                "final_asr": 0}