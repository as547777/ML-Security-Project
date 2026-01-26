from interfaces import AbstractAttack

class AttackHandler:
    def __init__(self,attack:AbstractAttack):
        self.attack=attack
    
    def handle(self, context):
        x_train = context["x_train"]
        y_train = context["y_train"]
        x_test = context["x_test"]
        y_test = context["y_test"]
        context["x_train_clean"]=x_train
        context["y_train_clean"]=y_train
        params = context.get("attack_params",{})
        params["dataset"] = context["dataset"]

        context["attack_instance"]=self.attack
        target_model = context.get('model')

        if hasattr(self.attack, 'skip_retraining'):
            context['skip_model_training'] = self.attack.skip_retraining

        (x_poisoned_train, y_poisoned_train, x_test_asr, y_test_asr) = self.attack.execute(target_model, (x_train, y_train, x_test, y_test), params)

        context["x_train"] = x_poisoned_train
        context["y_train"] = y_poisoned_train

        context["x_test_asr"] = x_test_asr
        context["y_test_asr"] = y_test_asr