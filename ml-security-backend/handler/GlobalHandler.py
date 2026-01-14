import copy
import random
import numpy as np
import torch

class GlobalHandler:
    def __init__(self, handlers=[], metric_handler=None):
        self.handlers=handlers
        self.metric_handler=metric_handler
    
    def register(self, handler):
        self.handlers.append(handler)
    
    def register_metric_handler(self, metric_handler):
        self.metric_handler=metric_handler
    

    def handle(self, context):
        num_runs=context.get("num_of_runs",1)
        seed_base=context.get("seed",42)

        context["accuracies"] = []
        context["accuracies_after"] = []
        context["asrs"] = []
        context["asrs_after"] = []

        for i in range(num_runs):
            seed=seed_base+i
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


            run_context = copy.deepcopy(context)

            for handler in self.handlers:
                    handler.handle(run_context)
            
            if i == 0:
                for key in ["w_res", "h_res", "color_channels", "classes"]:
                    context[key] = run_context[key]


            context["accuracies"].append(run_context.get("acc"))
            context["accuracies_after"].append(run_context.get("final_accuracy"))
            context["asrs"].append(run_context.get("acc_asr"))
            context["asrs_after"].append(run_context.get("final_asr"))
            
            del run_context

        self.metric_handler.handle(context)

        return context
        