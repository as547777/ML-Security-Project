from interfaces.AbstractMetric import AbstractMetric

class AttackIntensity(AbstractMetric):
    __desc__ = {
        "display_name": "AttackIntensity",
        "description": "Return poison rate and patch area ratio",
    }
    def compute(self, context):
        metrics=context.get("metrics",{})
        
        trigger_size=context["attack_params"]["trigger_size"]
        width = context['w_res']
        height = context['h_res']

        patch_area_ratio=(trigger_size*trigger_size)/(width*height)

        metrics["attack"]={
            "poison_rate":context["attack_params"]["poison_rate"],
            "patch_area_ratio":patch_area_ratio
        }
        context["metrics"]=metrics
