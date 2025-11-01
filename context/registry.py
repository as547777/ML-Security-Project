DATASET_MAP={}
DEFENSE_MAP={}
ATTACK_MAP={}
MODEL_MAP={}
METRIC_MAP={}


def register_attack(name, clss):
    ATTACK_MAP[name]=clss

def register_defense(name, clss):
    DEFENSE_MAP[name]=clss

def register_dataset(name, clss):
    DATASET_MAP[name]=clss

def register_model(name, clss):
    MODEL_MAP[name]=clss

def register_metric(name, clss):
    METRIC_MAP[name]=clss

