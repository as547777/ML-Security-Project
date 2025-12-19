import random
import torch
from utils.Visualizer import Visualizer

class VisualizationHandler:
    def __init__(self, num_samples=5):
        self.num_samples = num_samples

    def handle(self, context):
        x_test = context.get("x_test")
        y_test = context.get("y_test")
        model_wrapper = context.get("model")
        attack_instance = context.get("attack_instance")
        
        if x_test is None or model_wrapper is None:
            return

        model = model_wrapper.model
        model.eval()
        device = next(model.parameters()).device
        
        visualizations = []
        all_indices = list(range(len(x_test)))
        random.shuffle(all_indices)

        count = 0
        
        for idx in all_indices:
            if count >= self.num_samples:
                break
                
            original_img = x_test[idx].unsqueeze(0).to(device)
            original_label = y_test[idx].item()

            poisoned_img = attack_instance.apply_trigger(original_img.cpu().clone()).to(device)

            with torch.no_grad():
                pred_clean = model(original_img).argmax(dim=1).item()
                pred_poisoned = model(poisoned_img).argmax(dim=1).item()

            # Pokaži samo one gdje je napad zapravo promijenio predikciju u ciljanu
            if pred_poisoned == attack_instance.target_label and pred_clean != pred_poisoned:
                visualizations.append({
                    "original_image": Visualizer.tensor_to_base64(original_img[0]),
                    "poisoned_image": Visualizer.tensor_to_base64(poisoned_img[0]),
                    "original_label": int(original_label),
                    "prediction_clean": int(pred_clean),
                    "prediction_poisoned": int(pred_poisoned),
                    "target_label": attack_instance.target_label
                })
                count += 1

        context["visualizations"] = visualizations