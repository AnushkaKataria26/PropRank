import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from models.ranknet import RankNetMLP
from models.loss import ranknet_loss

logger = logging.getLogger(__name__)

class PairwiseDataset(Dataset):
    def __init__(self, pairs, weights=None):
        self.pairs = pairs
        self.weights = weights
        
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        item = {
            'item_i_features': torch.tensor(pair['item_i_features'], dtype=torch.float32),
            'item_j_features': torch.tensor(pair['item_j_features'], dtype=torch.float32),
            'label': torch.tensor(pair['label'], dtype=torch.float32)
        }
        if self.weights is not None:
            item['weight'] = torch.tensor(self.weights[idx], dtype=torch.float32)
        return item

def train_ranknet(train_pairs, config, propensity_weights=None):
    if not train_pairs:
        raise ValueError("train_pairs is empty")
        
    if propensity_weights is not None and len(propensity_weights) != len(train_pairs):
        raise ValueError(f"Length mismatch: {len(propensity_weights)} weights vs {len(train_pairs)} pairs")
        
    torch.manual_seed(config.random_seed)
    
    dataset = PairwiseDataset(train_pairs, weights=propensity_weights)
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
    
    input_dim = train_pairs[0]['item_i_features'].shape[0]
    model = RankNetMLP(input_dim=input_dim)
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    epoch_losses = []
    
    for epoch in range(config.train_epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            feat_i = batch['item_i_features']
            feat_j = batch['item_j_features']
            labels = batch['label']
            
            score_i = model(feat_i)
            score_j = model(feat_j)
            
            loss = ranknet_loss(score_i, score_j, labels, reduction='none')
            if 'weight' in batch:
                weight = batch['weight']
                if loss.dim() == 2 and weight.dim() == 1:
                    weight = weight.unsqueeze(1)
                loss = loss * weight
                
            loss = loss.mean()
            
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"Training halted: NaN or Inf loss detected at epoch {epoch + 1}.")
                
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
        mean_loss = total_loss / batches
        epoch_losses.append(mean_loss)
        logger.info(f"Epoch {epoch + 1}/{config.train_epochs} - Loss: {mean_loss:.4f}")
        
    return model, epoch_losses
