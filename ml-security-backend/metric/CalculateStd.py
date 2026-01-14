from interfaces.AbstractMetric import AbstractMetric
import math

class CalculateStd(AbstractMetric):
    __desc__ = {
        "display_name": "CalculateStd",
        "description": "Measure clean accuracy and asr mean and standard deviation over multiple runs",
    }

    def __init__(self, number_of_runs=3, start_seed=42):
        self.number_of_runs = number_of_runs
        self.start_seed = start_seed

    def calculate_mean_std(self,accs):
        mean = sum(accs) / len(accs)
        var = sum((x - mean)**2 for x in accs)/len(accs)
        std = math.sqrt(var)
        return mean, std

    def compute(self, context):
        metrics=context.get("metrics",{})

        clean_accs=context.get("accuracies",[])
        asrs=context.get("asrs",[])

        clean_accs_after=context.get("accuracies_after",[])
        asrs_after=context.get("asrs_after",[])

        acc_mean,acc_std = self.calculate_mean_std(clean_accs)
        acc_mean_after,acc_std_after=self.calculate_mean_std(clean_accs_after)
        asr_mean, asr_std = self.calculate_mean_std(asrs)
        asr_mean_after, asr_std_after=self.calculate_mean_std(asrs_after)

        metrics["deviations"] ={"clean_acc":{"before": {
            "mean": acc_mean,
            "std": acc_std
        }, "after":{
            "mean":acc_mean_after,
            "std":acc_std_after
        }},
        "asr":{
            "before":{
                "mean":asr_mean,
                "std":asr_std
            },"after":{
                "mean":asr_mean_after,
                "std":asr_std_after
            }
        }}
        context["metrics"]=metrics

