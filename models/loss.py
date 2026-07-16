import torch
import torch.nn.functional as F

def ranknet_loss(score_i, score_j, labels, reduction='mean'):
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.float32)
        
    # Validate labels are strictly 0.0 or 1.0 (or int equivalents)
    # Using torch.isclose or direct comparison
    if not torch.all((labels == 0.0) | (labels == 1.0)):
        raise ValueError("Labels must be strictly 0 or 1.")
        
    diff = score_i - score_j
    
    # Ensure shapes match for BCEWithLogitsLoss
    if labels.dim() == 1 and diff.dim() == 2:
        labels = labels.unsqueeze(1)
        
    loss = F.binary_cross_entropy_with_logits(diff, labels.float(), reduction=reduction)
    return loss
