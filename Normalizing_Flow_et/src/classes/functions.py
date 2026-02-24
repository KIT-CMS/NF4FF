import torch

def weighted_average(loss_per_sample, weights, eps = 1e-12):
    denom = weights.sum()
    denom = torch.clamp(denom, min=eps)
    return (loss_per_sample * weights).sum()/denom