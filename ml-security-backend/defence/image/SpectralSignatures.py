import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange
from interfaces.AbstractDeffense import AbstractDefense

class SpectralSignatures(AbstractDefense):
    __desc__ = {
        "name": "Spectral Signatures",
        "description": "A defense that detects and removes backdoor samples by analyzing feature covariance structure. It performs SVD on the feature representations and identifies outliers based on their projection onto the top principal components.",
        "type": "Defense",
        "params": {
            "percentile": {
                "label": "Outlier Percentile Threshold",
                "tooltip": "Percentile threshold for outlier detection. Samples with reconstruction scores above this percentile are considered suspicious and removed.",
                "type": "number",
                "step": 0.5,
                "value": 0.8
            },
            "n_components": {
                "label": "Number of Principal Components",
                "tooltip": "Number of top principal components (eigenvectors) to use for outlier detection.",
                "type": "number",
                "step": 1,
                "value": 1
            }
        }
    }
    
    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else "cpu")):
        self.device = device

    def execute(self, model, data, params, context):
        x_train_poisoned = context["x_train"]
        y_train_poisoned = context["y_train"]
        x_test = context["x_test"]
        y_test = context["y_test"]
        x_test_asr = context["x_test_asr"]
        y_test_asr = context["y_test_asr"]
        
        self.percentile = params.get("percentile", 0.8)
        self.n_components = params.get("n_components", 1)
        target_label = context["attack_params"]["target_label"]
        
        poisoned_trainset = []
        for i in range(len(x_train_poisoned)):
            poisoned_trainset.append((x_train_poisoned[i], y_train_poisoned[i]))
        
        filtered_poisoned_id, filtered_benign_id = self.filter(model.model, poisoned_trainset, target_label)
        
        x_train_filtered = x_train_poisoned[filtered_benign_id]
        y_train_filtered = y_train_poisoned[filtered_benign_id]

        lr = context["learning_rate"]
        momentum = context.get("train_params", {}).get("momentum", 0.9)
        epochs = context.get("train_params", {}).get("epochs", 10)
        
        init_params={"w_res": x_train_filtered.shape[2],"h_res": x_train_filtered.shape[3],
        "color_channels" : x_train_filtered.shape[1],
        "classes" : len(torch.unique(y_train_poisoned))}
        
        model.init(init_params)
        
        model.train((x_train_filtered, y_train_filtered), lr, momentum, epochs)
        
        
        _, clean_acc = model.predict((x_test, y_test))
        
        _, asr = model.predict((x_test_asr, y_test_asr))
        
        return {
            "final_accuracy": clean_acc,
            "final_asr": asr,
        }

    def filter(self, model, poisoned_trainset, lbl):
        poisoned_label = []
        for i in range(len(poisoned_trainset)):
            poisoned_label.append(poisoned_trainset[i][1])

        cur_indices = [i for i, v in enumerate(poisoned_label) if v==lbl]
        cur_examples = len(cur_indices)

        model.eval()

        for iex in trange(cur_examples):
            cur_im = cur_indices[iex]
            x_batch = poisoned_trainset[cur_im][0].unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = model.get_representations(x_batch)
                features_flat = features.cpu().numpy().flatten()
            
            if iex == 0:
                feature_dim = len(features_flat)
                full_cov = np.zeros(shape=(cur_examples, feature_dim))
            
            full_cov[iex] = features_flat
        
        full_mean = np.mean(full_cov, axis=0, keepdims=True)
        centered_cov = full_cov - full_mean
        u, s, v = np.linalg.svd(centered_cov, full_matrices=False)
    
        eigs = v[0:self.n_components]
        corrs = np.matmul(eigs, np.transpose(full_cov))
        scores = np.linalg.norm(corrs, axis=0)
        print(f'Length Scores: {len(scores)}')
        
        p_score = np.percentile(scores, self.percentile)
        top_scores = np.where(scores > p_score)[0]
        
        filtered_poisoned_id = np.copy(top_scores)
        print('removed_inds_length:' + str(len(filtered_poisoned_id)))

        re = [cur_indices[v] for i, v in enumerate(filtered_poisoned_id)]
        filtered_benign_id = np.delete(range(len(poisoned_trainset)), re)
        print('left_inds_length:' + str(len(filtered_benign_id)))   

        return filtered_poisoned_id, filtered_benign_id







