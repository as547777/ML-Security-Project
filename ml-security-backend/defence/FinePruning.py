import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np
from interfaces.AbstractDeffense import AbstractDefense

'''
Obrana funckionira na način da ima dva koraka:

1. pruning - prolazimo kroz neuronsku mrežu te mičemo (postavljamo na nulu) one neurone koji su najmanje važni za normalnu klasifikaciju
-> imaju najmanje težine
2. fine tuning - nakon pruninga, točnost modela na čistim podacima opada pa treniramo model s malim learning rateom 
kako bismo popravili točnost -> backdoor ostaje uništen 
'''

class FinePruning(AbstractDefense):
    def __init__(self, target_layer_name = 'conv2'):
        self.target_layer_name = target_layer_name #sloj nad kojim ćemo obaviti pruning

    def prune_model(self, torch_model, prune_rate):
        '''
        pruning one koji imaju najmanju L1 normu (težinu)
        '''
        conv_layers = []
        # Prolazimo kroz sve module modela i spremamo one koji su Conv2d
        for m in torch_model.modules():
            if isinstance(m, nn.Conv2d):
                conv_layers.append(m)

        module = conv_layers[-1]
        
        weights = module.weight.data.abs().clone()
        l1_norm = weights.view(weights.size(0), -1).sum(dim=1) #L1 norma

        num_filters = weights.shape[0]
        num_prune = int(num_filters * prune_rate)
        
        #pronađi indekse s najmanjom normom
        _, indices = torch.topk(l1_norm, k=num_prune, largest=False)

        module.weight.data[indices] = 0 #postavi te težine na 0
        
        if module.bias is not None:
            module.bias.data[indices] = 0 #i bias na 0

        return torch_model
    
    def evaluate(self, model, data_clean, data_poison):
        _, acc_clean = model.predict(data_clean)
        _, acc_asr = model.predict(data_poison)
        return acc_clean, acc_asr
    
    def execute(self, model, data, params):
        x_train, y_train, x_test, y_test = data

        x_test_asr = params.get("x_test_asr")
        y_test_asr = params.get("y_test_asr")

        prune_rate = params.get("prune_rate", 0.2)
        ft_epochs = params.get("fine_tune_epochs", 5)
        lr = params.get("learning_rate", 0.01)

        torch_model = model.model 
        torch_model = self.prune_model(torch_model, prune_rate)

        acc_clean_pruned, acc_asr_pruned = self.evaluate(
            model, (x_test, y_test), (x_test_asr, y_test_asr)
        )

        #fine tuning
        model.train((x_train, y_train), lr=lr, momentum=0.9, epochs=ft_epochs)

        # Evaluacija nakon fine-tuninga
        acc_clean_final, acc_asr_final = self.evaluate(
            model, (x_test, y_test), (x_test_asr, y_test_asr)
        )

        return {
            "pruned_accuracy": acc_clean_pruned,
            "pruned_asr": acc_asr_pruned,
            "final_accuracy": acc_clean_final,
            "final_asr": acc_asr_final
        }