from interfaces.AbstractMetric import AbstractMetric

class AccuracyDifference(AbstractMetric):
    __desc__ = {
        "display_name": "AccuracyDifference",
        "description": "Measure clean accuracy and asr difference before and after backdoor defense is applied",
    }
    def compute(self, context):
        metrics=context.get("metrics",{})
        
        accs = context.get("accuracies", [])
        accs_after = context.get("accuracies_after", [])
        asrs = context.get("asrs", [])
        asrs_after = context.get("asrs_after", [])

        acc_drops = [acc - acc_after for acc, acc_after in zip(accs, accs_after)]
        asr_reductions = [asr - asr_a for asr, asr_a in zip(asrs, asrs_after)]

        metrics["improvement"] = {
            "acc_drop": acc_drops,
            "asr_reduction": asr_reductions
        }

        context["metrics"] = metrics

