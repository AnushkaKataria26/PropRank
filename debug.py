import torch
from models.train import train_ranknet, PairwiseDataset
import numpy as np

def run_debug():
    pairs = []
    for i in range(10):
        pairs.append({
            'item_i_features': np.array([1.0]),
            'item_j_features': np.array([0.0]),
            'label': 1.0
        })
    class Config:
        train_epochs = 1
        learning_rate = 0.1
        random_seed = 42
        train_batch_size = 4
    config = Config()
    
    # baseline
    m1, _ = train_ranknet(pairs, config)
    w1 = list(m1.parameters())[0].clone()
    
    # corrected
    m2, _ = train_ranknet(pairs, config, propensity_weights=np.array([2.0]*10))
    w2 = list(m2.parameters())[0].clone()
    
    print("w1:", w1)
    print("w2:", w2)
    print("Equal:", torch.allclose(w1, w2))
    
if __name__ == '__main__':
    run_debug()
