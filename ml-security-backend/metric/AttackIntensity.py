from interfaces.AbstractMetric import AbstractMetric

class AttackIntensity(AbstractMetric):
    __desc__ = {
        "display_name": "AttackIntensity",
        "description": "Return poison rate and patch area ratio",
    }
    def compute(self, context):
        metrics=context.get("metrics",{})
        
        attack_params=context.get("attack_params",{})
        trigger_size=attack_params.get("trigger_size")

        attack_metrics={
            "poison_rate": attack_params.get("poison_rate")
        }

        if trigger_size is not None:
            width = context['w_res']
            height = context['h_res']

            patch_area_ratio=(trigger_size*trigger_size)/(width*height)
            attack_metrics["patch_area_ratio"]=patch_area_ratio
            
        metrics["attack"]=attack_metrics
        context["metrics"]=metrics
