import torch
from interfaces.AbstractDeffense import AbstractDefense
from model.ImageModel import ImageModel
import numpy as np

class SpectralSignatures(AbstractDefense):
    __desc__ = {
        "name": "Spectral Signatures",
        "description": "A defense that detects and removes backdoor samples by analyzing feature covariance structure. For each class, it performs SVD on the feature representations and identifies outliers based on their projection onto the top principal components.",
        "type": "Defense",
        "params": {
            "percentile": {
                "label": "Outlier Percentile Threshold",
                "tooltip": "Percentile threshold for outlier detection. Samples with reconstruction scores above this percentile are considered suspicious and removed.",
                "type": "number",
                "step": 0.5,
                "value": 99
            },
            "n_components": {
                "label": "Number of Principal Components",
                "tooltip": "Number of top principal components (eigenvectors) to use for outlier detection.",
                "type": "number",
                "step": 1,
                "value": 5
            }
        }
    }
    def __init__(self,device=torch.device('cuda' if torch.cuda.is_available() else "cpu")):
        self.device=device

    def execute(self, model,data, params,context):
        x_train_poisoned=context["x_train"]
        y_train_poisoned=context["y_train"]
      
        x_test = context["x_test"]
        y_test = context["y_test"]

        x_test_asr = context["x_test_asr"]
        y_test_asr = context["y_test_asr"]

        _, acc = model.predict((x_test, y_test))
        _, acc_asr = model.predict((x_test_asr, y_test_asr))
       
        context["acc_before"]=acc
        context["asr_before"]=acc_asr
        
        percentile = params.get("percentile", 99)
        n_components = params.get("n_components", 5)

        self.percentile = percentile
        self.n_components = n_components

        suspicious_indices=self.detect(model,x_train_poisoned,y_train_poisoned)
        
        x_cleaned,y_cleaned=self.remove_indices(x_train_poisoned,y_train_poisoned,suspicious_indices)

        model.train((x_cleaned,y_cleaned),context["learning_rate"],context["momentum"],context["epochs"])

        _, acc_after = model.predict((x_test, y_test))
        _, acc_asr_after = model.predict((x_test_asr, y_test_asr))

        context["acc_after"] = acc_after
        context["asr_after"] = acc_asr_after

        return {"final_accuracy": acc_after,
            "final_asr": acc_asr_after}

    
    @torch.no_grad()
    def detect(self, model, x_train,y_train):
        model.model.eval()

        x_train=x_train.to(self.device)
        y_train=y_train.to(self.device)

        suspiciouos_indices=[]

        labels=torch.unique(y_train)

        for label in labels:
            class_indices=torch.where(y_train==label)[0]

            rep=model.model.get_representations(x_train[class_indices])
            rep=rep.reshape(rep.size(0),-1)

            cov = rep.cpu().numpy()

            cov_mean=np.mean(cov,axis=0,keepdims=True)

            cov_centered=cov-cov_mean

            u,s,v=np.linalg.svd(cov_centered,full_matrices=False)
            eigs=v[0:self.n_components]

            corrs=np.matmul(eigs,np.transpose(cov_centered))
            scores=np.linalg.norm(corrs,axis=0)
            
            p_score=np.percentile(scores,self.percentile)
            to_remove=np.where(scores>p_score)[0]

            remove_inds=class_indices[to_remove].cpu().numpy()
            suspiciouos_indices.extend(remove_inds)
        
        return np.array(suspiciouos_indices)
    
    def remove_indices(self,x,y,remove_indices):
        mask=torch.ones(len(x),dtype=torch.bool)
        mask[remove_indices]=False
        return x[mask],y[mask]

