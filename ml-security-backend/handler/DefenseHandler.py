from interfaces import AbstractDeffense

class DefenseHandler:
    def __init__(self, defense:AbstractDeffense):
        self.defense=defense

    def handle(self, context):
        model_handler = context.get("model_handler")

        model = context["model"]

        x_train = context["x_train"]
        y_train = context["y_train"]
        x_test = context["x_test"]
        y_test = context["y_test"]

        x_test_asr = context.get("x_test_asr")
        y_test_asr = context.get("y_test_asr")

        defense_params = context.get("defense_params", {})
        defense_params["x_test_asr"] = x_test_asr
        defense_params["y_test_asr"] = y_test_asr

        original_epochs = context.get("epochs", 10) #broj epoha za fine-tuning
        defense_params["fine_tune_epochs"] = max(1, int(original_epochs * 0.1))
        defense_params["learning_rate"] = context.get("learning_rate", 0.01)

        # Izvrši obranu
        results = self.defense.execute(model, (x_train, y_train, x_test, y_test), defense_params,context)

        context["defense_results"] = results

        context["acc_clean_pruned"] = results.get("pruned_accuracy","")
        context["acc_asr_pruned"] = results.get("pruned_asr","")

        context["final_accuracy"] = results["final_accuracy"]
        context["final_asr"] = results["final_asr"]